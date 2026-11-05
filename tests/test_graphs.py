"""T9 checks that need no GPU: the decode step cut into pieces at the routers (run eagerly on a CPU).

* The pieces with the fixed-size, masked key/value cache reproduce the eager free-running decode of
  ``interleaved_freerun``: the same tokens, routing at every event, transfers and logits up to float rounding.
* The runner refuses prefetching and host-resident experts.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for relative in ("src/runtime", "src", "analysis"):
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
class GraphedDecodeTests(unittest.TestCase):
    STEPS = 14

    def _model(self):
        from transformers import OlmoeConfig, OlmoeForCausalLM

        config = OlmoeConfig(hidden_size=32, intermediate_size=16, num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, vocab_size=64, max_position_embeddings=64,
                             num_experts=8, num_experts_per_tok=2)
        torch.manual_seed(0)
        return OlmoeForCausalLM(config).eval().float()

    def _greedy_trace(self, model, arch, prompt):
        from offload_runtime import OffloadRuntime

        runtime = OffloadRuntime(model, {"architecture": arch}, torch.device("cpu"), None, 8)
        tokens, requests = [], []
        runtime.mode = "prefill"
        with torch.inference_mode():
            out = model(input_ids=prompt, use_cache=True, return_dict=True)
            past = out.past_key_values
            token = int(out.logits[0, -1].float().argmax())
            runtime.mode = "decode"
            for _ in range(self.STEPS):
                tokens.append(token)
                runtime.current_token = token
                runtime.reset_token_counters()
                runtime.routing = []
                out = model(input_ids=torch.tensor([[token]]), past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                requests.append([sorted(int(e) for e in r[1].tolist()) for r in sorted(runtime.routing, key=lambda r: r[0])])
                token = int(out.logits[0, -1].float().argmax())
        runtime.release()
        return tokens, requests

    def _run(self, model, arch, prompt, tokens, requests, runtime_class, run_path, experts_in_graphs=None):
        from policy import SimplePolicy

        class Tokenizer:
            eos_token_id = None

            def __call__(self, text, **kwargs):
                return {"input_ids": prompt.clone(), "attention_mask": torch.ones_like(prompt)}

        spec = {"architecture": arch, "trace_generation": {"max_prompt_tokens": 512, "stop_on_eos": True},
                "sequences": [{"sequence_id": 0, "prompt": "p", "prompt_length": int(prompt.shape[1]),
                               "generated_token_ids": list(tokens),
                               "trace_requested_expert_ids": np.asarray(requests).reshape(-1).tolist()}]}
        workload = {"name": "test", "regime": "test",
                    "sequences": [{"position": 0, "source_sequence_id": 0, "domain": "general"}]}
        policy = SimplePolicy("lfu_decay", 2, 8, 4, 2)
        runtime = runtime_class(model, spec, torch.device("cpu"), policy, 4)
        if experts_in_graphs is not None:
            runtime.graphed_decoder.experts_in_graphs = experts_in_graphs
        try:
            result = run_path(runtime, Tokenizer(), spec, workload, teacher_force=True, limit_sequences=None,
                              log=lambda message: None)
        finally:
            runtime.release()
        return result["sequences"][0], runtime

    def test_pieces_reproduce_the_eager_free_running_decode(self) -> None:
        import interleaved_graphs
        from interleaved_freerun import free_running_run_path
        from offload_runtime import OffloadRuntime

        model = self._model()
        arch = {"layer_indices": [0, 1], "num_moe_layers": 2, "num_experts": 8, "top_k": 2, "expert_bytes": 3 * 32 * 16 * 4}
        torch.manual_seed(3)
        prompt = torch.randint(0, 64, (1, 6))
        tokens, requests = self._greedy_trace(model, arch, prompt)
        eager, _ = self._run(model, arch, prompt, tokens, requests, OffloadRuntime, free_running_run_path)
        graphed, runtime = self._run(model, arch, prompt, tokens, requests, interleaved_graphs.GraphedRuntime,
                                     interleaved_graphs.graphed_run_path)
        self.assertFalse(runtime.graphed_decoder.graphed)  # a CPU runs the pieces eagerly
        self.assertFalse(runtime.graphed_decoder.experts_in_graphs)
        self.assertEqual(runtime.graphed_decoder.max_len, int(prompt.shape[1]) + self.STEPS + 1)
        inside, _ = self._run(model, arch, prompt, tokens, requests, interleaved_graphs.GraphedRuntime,
                              interleaved_graphs.graphed_run_path, experts_in_graphs=True)  # the experts module in the pieces
        for row in (eager, graphed, inside):
            self.assertEqual(row["tokens"], self.STEPS)
            self.assertEqual(row["routing_matches"], row["routing_events"])
            self.assertEqual(row["token_agreements"], row["token_comparisons"])
            self.assertTrue(row["first_token_agrees"])
        self.assertEqual(graphed["expert_transfers"], eager["expert_transfers"])
        self.assertEqual(inside["expert_transfers"], eager["expert_transfers"])
        self.assertEqual(inside["generated_tokens"], graphed["generated_tokens"])
        self.assertGreater(graphed["expert_transfers"], 0)

    def test_pieces_match_the_eager_logits(self) -> None:
        """Logits of every piece-wise decode step against the eager model with its growing cache."""

        import interleaved_graphs
        from policy import SimplePolicy

        model = self._model()
        arch = {"layer_indices": [0, 1], "num_moe_layers": 2, "num_experts": 8, "top_k": 2, "expert_bytes": 3 * 32 * 16 * 4}
        torch.manual_seed(4)
        prompt = torch.randint(0, 64, (1, 5))
        steps = torch.randint(0, 64, (10,)).tolist()
        spec = {"architecture": arch, "sequences": [{"prompt_length": 5, "generated_token_ids": steps}]}
        runtime = interleaved_graphs.GraphedRuntime(model, spec, torch.device("cpu"), SimplePolicy("lru", 2, 8, 4, 2), 4)
        decoder = runtime.graphed_decoder
        reference = interleaved_graphs.GraphedRuntime(model, spec, torch.device("cpu"), SimplePolicy("lru", 2, 8, 4, 2), 4)
        with torch.inference_mode():
            runtime.mode = reference.mode = "prefill"
            out = model(input_ids=prompt, use_cache=True, return_dict=True)
            decoder.begin_sequence(out.past_key_values, 5)
            past = out.past_key_values
            for position, token in enumerate(steps, start=5):
                for rt in (runtime, reference):
                    rt.current_token = int(token)
                    rt.reset_token_counters()
                    rt.routing = []
                decoder.step(token, position)
                logits = model.lm_head(model.model.norm(decoder.layer_output(decoder.L - 1)))[0, -1]
                reference.mode = "decode"
                expected = model(input_ids=torch.tensor([[token]]), past_key_values=past, use_cache=True,
                                 return_dict=True)
                past = expected.past_key_values
                self.assertTrue(torch.allclose(logits, expected.logits[0, -1], atol=1e-5, rtol=1e-5))
                self.assertEqual([r[1].tolist() for r in runtime.routing], [r[1].tolist() for r in reference.routing])
                reference.mode = "prefill"
        reference.release()  # built second, so its saved forward is the first runtime's: release it first
        runtime.release()

    def test_runner_refuses_prefetch_and_host_experts(self) -> None:
        import interleaved_graphs

        for argv in (["--experts-on-host", "--output", "x.json"], ["--arms", "lfu_decay+pf", "--output", "x.json"]):
            with self.assertRaises(SystemExit):
                interleaved_graphs.main(argv)


if __name__ == "__main__":
    unittest.main()
