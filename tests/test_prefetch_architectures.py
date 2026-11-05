"""Prefetching with experts kept on the host, on tiny OLMoE, Qwen3-MoE and Mixtral models (T5).

``test_hardware.py`` (``PrefetchRuntimeTests``) checks the prefetcher on OLMoE with a resident reference. The
H100 runs prefetch on Qwen3 with every expert in pinned host memory, and run HOBBIT's rule with prefetching.
Here, for each architecture, decoding with the prefetcher (g1 and g1n guesses; decayed LFU and HOBBIT's rule)
produces the resident model's logits and routing, and prefetches happen.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for relative in ("src/runtime", "src"):
    value = str(ROOT / relative)
    if value not in sys.path:
        sys.path.append(value)

try:
    import torch
    import transformers  # noqa: F401

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch/transformers missing")
class PrefetchArchitectureTest(unittest.TestCase):
    KINDS = ("olmoe", "qwen3", "mixtral")

    def _model(self, kind: str):
        import transformers as T

        common = dict(hidden_size=32, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                      vocab_size=64, max_position_embeddings=64, num_experts_per_tok=2)
        if kind == "olmoe":
            config, cls = T.OlmoeConfig(intermediate_size=16, num_experts=8, **common), T.OlmoeForCausalLM
        elif kind == "qwen3":
            config = T.Qwen3MoeConfig(moe_intermediate_size=16, intermediate_size=48, num_experts=8, head_dim=8,
                                      decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True, **common)
            cls = T.Qwen3MoeForCausalLM
        else:
            config, cls = T.MixtralConfig(intermediate_size=16, num_local_experts=8, **common), T.MixtralForCausalLM
        torch.manual_seed(0)
        model = cls(config).eval().float()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.dim() >= 2:
                    param.normal_(0.0, 0.3)
                elif "layernorm" in name:  # distinct RMSNorm gains, so g1n differs from g1
                    param.uniform_(0.5, 1.5)
        return model

    def _spec(self):
        return {"architecture": {"layer_indices": [0, 1, 2], "num_moe_layers": 3, "num_experts": 8, "top_k": 2,
                                 "expert_bytes": 3 * 32 * 16 * 4}}

    def _device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _decode(self, model, runtime, device, prompt, steps):
        runtime.mode = "prefill"
        with torch.inference_mode():
            out = model(input_ids=prompt.to(device), use_cache=True, return_dict=True)
            past, logits, routing = out.past_key_values, [out.logits[0, -1].float().cpu()], []
            runtime.mode = "decode"
            for token in steps:
                runtime.current_token = int(token)
                runtime.reset_token_counters()
                runtime.routing = []
                out = model(input_ids=torch.tensor([[token]], device=device), past_key_values=past,
                            use_cache=True, return_dict=True)
                past = out.past_key_values
                logits.append(out.logits[0, -1].float().cpu())
                routing.append([(o, r.tolist()) for o, r, _g in runtime.routing])
        return torch.stack(logits), routing

    def test_prefetch_with_host_experts_keeps_the_output(self) -> None:
        from offload_runtime import OffloadRuntime, PrefetchConfig, place_non_experts
        from policy import SimplePolicy
        from policy_baselines import HobbitPolicy

        # float32 against the resident model, on the CPU like test_hardware's prefetch test: the CUDA guess path
        # reads bf16 logits (the runs are bf16) and is covered by the bf16 test below
        device = torch.device("cpu")
        torch.manual_seed(1)
        prompt = torch.randint(0, 64, (1, 7))
        steps = torch.randint(0, 64, (16,)).tolist()
        for kind in self.KINDS:
            base = self._model(kind)
            resident_model = copy.deepcopy(base).to(device)
            resident = OffloadRuntime(resident_model, self._spec(), device, None, 8)
            reference, routing = self._decode(resident_model, resident, device, prompt, steps)
            resident.release()
            for variant in ("g1", "g1n"):
                for make in (lambda: SimplePolicy("lfu_decay", 3, 8, 4, 2), lambda: HobbitPolicy(3, 8, 4, 2, w=0.2)):
                    policy = make()
                    with self.subTest(kind=kind, variant=variant, policy=policy.name):
                        hosted = copy.deepcopy(base)
                        place_non_experts(hosted, device, [0, 1, 2])
                        runtime = OffloadRuntime(hosted, self._spec(), device, policy, 4, experts_on_host=True,
                                                 prefetch=PrefetchConfig(m=4, q=2, variant=variant))
                        measured, routing_hosted = self._decode(hosted, runtime, device, prompt, steps)
                        runtime.release()
                        self.assertTrue(torch.allclose(reference, measured, atol=1e-4, rtol=1e-4),
                                        f"{kind}: max diff {(reference - measured).abs().max().item():.2e}")
                        self.assertEqual(routing, routing_hosted)
                        self.assertGreater(policy.prefetch_transfers, 0)


    @unittest.skipUnless(HAVE_TORCH and torch.cuda.is_available(), "needs CUDA")
    def test_prefetch_with_host_experts_runs_on_cuda_in_bf16(self) -> None:
        """The H100 runs' path: bf16 weights, experts pinned on the host, driver-API copies, asynchronous guesses."""

        from offload_runtime import OffloadRuntime, PrefetchConfig, place_non_experts
        from policy import SimplePolicy
        from policy_baselines import HobbitPolicy

        device = torch.device("cuda")
        torch.manual_seed(3)
        prompt = torch.randint(0, 64, (1, 7))
        steps = torch.randint(0, 64, (12,)).tolist()
        for kind in self.KINDS:
            for make in (lambda: SimplePolicy("lfu_decay", 3, 8, 4, 2), lambda: HobbitPolicy(3, 8, 4, 2, w=0.2)):
                policy = make()
                with self.subTest(kind=kind, policy=policy.name):
                    model = self._model(kind).to(torch.bfloat16)
                    place_non_experts(model, device, [0, 1, 2])
                    runtime = OffloadRuntime(model, self._spec(), device, policy, 4, experts_on_host=True,
                                             prefetch=PrefetchConfig(m=4, q=2, variant="g1n"), copy_api="driver")
                    logits, routing = self._decode(model, runtime, device, prompt, steps)
                    runtime.release()
                    self.assertTrue(bool(torch.isfinite(logits).all()))
                    self.assertEqual(len(routing), len(steps))
                    self.assertGreater(policy.prefetch_transfers, 0)
                    self.assertGreater(policy.transfers, 0)

    def test_generator_capture_matches_the_offloaded_decode_on_all_architectures(self) -> None:
        """From T5 on, every new trace is generated with the offloaded runtime (fresh_trace_offloaded.py without
        --resident), OLMoE included: the trace generator's routing capture records exactly the requests the runtime
        served."""

        opt = next((p for p in (ROOT / "measurement",)
                    if (p / "stage3_residency/src/residency_headroom").exists()), None)
        if opt is None:
            self.skipTest("measurement code missing")
        for relative in ("src", "stage3_residency/src", "stage3_residency/stage1_prediction/src"):
            value = str(opt / relative)
            if value not in sys.path:
                sys.path.insert(0, value)
        from expert_analysis.modeling import discover_moe_layers
        from residency_headroom.trace_generation import DecodeRoutingCapture

        from offload_runtime import OffloadRuntime, place_non_experts
        from policy import SimplePolicy

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        torch.manual_seed(4)
        prompt = torch.randint(0, 64, (1, 6)).to(device)
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                model = self._model(kind)
                place_non_experts(model, device, [0, 1, 2])
                runtime = OffloadRuntime(model, self._spec(), device, SimplePolicy("lfu_decay", 3, 8, 4, 2), 4,
                                         experts_on_host=True)
                runtime.mode = "auto"
                specs = discover_moe_layers(model)
                self.assertEqual(len(specs), 3)
                with torch.inference_mode(), DecodeRoutingCapture(specs) as capture:
                    out = model(input_ids=prompt, use_cache=True, return_dict=True)
                    past = out.past_key_values
                    for _ in range(8):
                        token = torch.argmax(out.logits[:, -1, :].float(), dim=-1)
                        runtime.routing = []
                        capture.begin_token()
                        out = model(input_ids=token[:, None], past_key_values=past, use_cache=True, return_dict=True)
                        records = capture.finish_token()
                        past = out.past_key_values
                        self.assertEqual([sorted(r.expert_ids.tolist()) for r in records],
                                         [sorted(req.tolist()) for _o, req, _g in runtime.routing])
                runtime.release()

    def test_repeated_copies_change_no_output(self) -> None:
        """``interleaved_h100.py --copy-repeat K`` issues every expert copy K times into its slot: same output."""

        from offload_runtime import OffloadedMoeLayer, OffloadRuntime, place_non_experts
        from policy import SimplePolicy

        device = self._device()
        torch.manual_seed(2)
        prompt = torch.randint(0, 64, (1, 5))
        steps = torch.randint(0, 64, (10,)).tolist()
        base = self._model("qwen3")
        resident_model = copy.deepcopy(base).to(device)
        resident = OffloadRuntime(resident_model, self._spec(), device, None, 8)
        reference, routing = self._decode(resident_model, resident, device, prompt, steps)
        resident.release()
        single = OffloadedMoeLayer.copy_expert
        calls = []

        def repeated(layer, expert, slot, stream_handle, _k=3):
            for _ in range(_k):
                calls.append((expert, slot))
                single(layer, expert, slot, stream_handle)

        OffloadedMoeLayer.copy_expert = repeated
        try:
            hosted = copy.deepcopy(base)
            place_non_experts(hosted, device, [0, 1, 2])
            policy = SimplePolicy("lfu_decay", 3, 8, 4, 2)
            runtime = OffloadRuntime(hosted, self._spec(), device, policy, 4, experts_on_host=True)
            measured, routing_hosted = self._decode(hosted, runtime, device, prompt, steps)
            runtime.release()
        finally:
            OffloadedMoeLayer.copy_expert = single
        self.assertTrue(torch.allclose(reference, measured, atol=1e-4, rtol=1e-4))
        self.assertEqual(routing, routing_hosted)
        self.assertGreater(policy.transfers, 0)
        self.assertEqual(len(calls) % 3, 0)


if __name__ == "__main__":
    unittest.main()
