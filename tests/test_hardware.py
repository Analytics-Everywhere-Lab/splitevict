"""Hardware-path checks that need no GPU and no model weights.

* ``FlatForest`` / ``NumbaForest`` reproduce scikit-learn's predictions.
* The simple policies reproduce the frozen ``residency_headroom`` policies on a random
  request stream.
* The offloaded MoE forward (slots filled from host copies) produces the same hidden
  states as the resident forward on a tiny random OLMoE, and its transfer count equals
  an LRU replay of the routing it observed.

Learned-policy exactness against the simulator is checked by
``src/runtime/replay_overhead.py`` (``results/a40/replay_validation_b32.json``).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for relative in ("src/runtime", "src"):
    value = str(ROOT / relative)
    if value not in sys.path:
        sys.path.append(value)

try:
    import sklearn  # noqa: F401

    HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    HAVE_SKLEARN = False
try:
    import numba  # noqa: F401

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False
try:
    import torch
    import transformers  # noqa: F401

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

OPT = Path(__file__).resolve().parents[1] / "measurement"


@unittest.skipUnless(HAVE_SKLEARN, "scikit-learn missing")
class FastTreeTests(unittest.TestCase):
    def setUp(self) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        rng = np.random.default_rng(1)
        x = rng.normal(size=(4000, 7))
        x[rng.random(x.shape) < 0.02] = np.nan
        y = np.nan_to_num(x[:, 0] - 0.5 * x[:, 2] ** 2 + rng.normal(scale=0.1, size=4000))
        self.x = x
        self.reg = HistGradientBoostingRegressor(max_iter=40, max_leaf_nodes=15, random_state=0).fit(x, y)
        self.clf = HistGradientBoostingClassifier(max_iter=30, max_leaf_nodes=7, random_state=0).fit(x, (y > 0).astype(int))

    def test_flat_forest_matches_sklearn(self) -> None:
        from fast_trees import FlatForest

        rows = self.x[:50]
        reg = FlatForest.from_sklearn(self.reg)
        clf = FlatForest.from_sklearn(self.clf)
        np.testing.assert_allclose(reg.predict(rows), self.reg.predict(rows), rtol=0, atol=1e-10)
        np.testing.assert_allclose(clf.predict_proba_positive(rows), self.clf.predict_proba(rows)[:, 1], rtol=0, atol=1e-10)

    @unittest.skipUnless(HAVE_NUMBA, "numba missing")
    def test_numba_forest_matches_sklearn(self) -> None:
        from fast_trees import FlatForest

        rows = self.x[:50]
        reg = FlatForest.from_sklearn(self.reg).to_numba()
        clf = FlatForest.from_sklearn(self.clf).to_numba()
        np.testing.assert_allclose(reg.predict(rows), self.reg.predict(rows), rtol=0, atol=1e-10)
        np.testing.assert_allclose(clf.predict_proba_positive(rows), self.clf.predict_proba(rows)[:, 1], rtol=0, atol=1e-10)


@unittest.skipUnless((OPT / "stage3_residency/src/residency_headroom/policies.py").exists(), "measurement code missing")
class SimplePolicyTests(unittest.TestCase):
    def test_simple_policies_match_frozen_definitions(self) -> None:
        sys.path.insert(0, str(OPT / "stage3_residency/src"))
        from residency_headroom.policies import make_policy as frozen_policy

        from policy import SimplePolicy

        rng = np.random.default_rng(3)
        layers, experts, top_k, capacity = 3, 16, 4, 7
        static = rng.random((layers, experts))
        for name in ("lru", "lfu", "lfu_decay", "static_hotset"):
            mine = SimplePolicy(name, layers, experts, capacity, top_k, static_scores=static)
            theirs = [
                frozen_policy(name, capacity, experts, alpha=0.95, static_scores=static[layer])
                for layer in range(layers)
            ]
            misses = 0
            for _ in range(400):
                layer = int(rng.integers(layers))
                request = rng.choice(experts, size=top_k, replace=False)
                result = mine.step(layer, 0, request, np.ones(top_k))
                transition = theirs[layer].process(request.tolist())
                misses += len(transition.misses)
                self.assertEqual(result.misses, len(transition.misses), name)
                self.assertEqual(set(mine.resident_experts(layer).tolist()), set(transition.after), name)
            self.assertEqual(mine.transfers, misses, name)


@unittest.skipUnless(HAVE_TORCH, "torch/transformers missing")
class OffloadedForwardTests(unittest.TestCase):
    def _tiny_model(self):
        from transformers import OlmoeConfig, OlmoeForCausalLM

        config = OlmoeConfig(hidden_size=32, intermediate_size=16, num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, vocab_size=64, max_position_embeddings=64,
                             num_experts=8, num_experts_per_tok=2)
        torch.manual_seed(0)
        return OlmoeForCausalLM(config).eval().float()

    def _spec(self):
        return {"architecture": {"layer_indices": [0, 1], "num_moe_layers": 2, "num_experts": 8, "top_k": 2,
                                 "expert_bytes": 3 * 32 * 16 * 4}}

    def test_offloaded_forward_equals_resident_and_counts_lru_transfers(self) -> None:
        from offload_runtime import OffloadRuntime
        from policy import SimplePolicy

        model = self._tiny_model()
        spec = self._spec()
        device = torch.device("cpu")
        torch.manual_seed(1)
        prompt = torch.randint(0, 64, (1, 6))
        steps = torch.randint(0, 64, (12,)).tolist()

        def decode(runtime):
            runtime.mode = "prefill"
            with torch.inference_mode():
                out = model(input_ids=prompt, use_cache=True, return_dict=True)
                past = out.past_key_values
                logits = []
                routing = []
                runtime.mode = "decode"
                for token in steps:
                    runtime.current_token = int(token)
                    runtime.reset_token_counters()
                    runtime.routing = []
                    out = model(input_ids=torch.tensor([[token]]), past_key_values=past, use_cache=True, return_dict=True)
                    past = out.past_key_values
                    logits.append(out.logits[0, -1].clone())
                    routing.append(list(runtime.routing))
            return torch.stack(logits), routing

        resident = OffloadRuntime(model, spec, device, None, 8)
        reference, routing = decode(resident)
        resident.release()

        policy = SimplePolicy("lru", 2, 8, 4, 2)
        offloaded = OffloadRuntime(model, spec, device, policy, 4)
        measured, routing_offloaded = decode(offloaded)
        offloaded.release()
        self.assertTrue(torch.allclose(reference, measured, atol=1e-5, rtol=1e-5), "offloaded hidden states differ from resident")
        self.assertEqual([[r[0] for r in s] for s in routing], [[r[0] for r in s] for s in routing_offloaded])

        replay = SimplePolicy("lru", 2, 8, 4, 2)
        for token_routing in routing:
            for ordinal, request, gates in token_routing:
                replay.step(ordinal, 0, request, gates)
        self.assertEqual(policy.transfers, replay.transfers)
        self.assertGreater(policy.transfers, 0)

        # the slots hold exactly the resident experts, and their weights are the originals
        for layer in offloaded.layers:
            resident_ids = policy.resident_experts(layer.ordinal)
            self.assertEqual(sorted(int(e) for e in np.flatnonzero(layer.slot_of >= 0)), sorted(resident_ids.tolist()))
            for expert in resident_ids:
                slot = int(layer.slot_of[expert])
                self.assertTrue(torch.equal(layer.slot_gate_up[slot], layer.resident_gate_up[int(expert)]))
                self.assertTrue(torch.equal(layer.slot_down[slot], layer.resident_down[int(expert)]))



BUNDLES = ROOT / "data" / "rankers"


@unittest.skipUnless(HAVE_SKLEARN and HAVE_NUMBA and (OPT / "stage3_residency").exists()
                     and (BUNDLES / "olmoe_dev_lag_req_trees200.pkl").exists(),
                     "needs the measurement data and the development split-phase rankers")
class SplitPhaseServiceTests(unittest.TestCase):
    """The running service (one worker, shared memory) makes the simulator's decisions exactly."""

    def test_service_matches_simulation_on_two_sequences(self) -> None:
        sys.path.insert(0, str(ROOT / "analysis"))
        from splitphase import SplitPhase, load_bundle, load_dev
        from splitphase_service import SplitPhasePolicy, SplitPhaseService

        inputs, _fit, deploy = load_dev()
        trace = inputs.trace
        p2 = str(BUNDLES / "olmoe_dev_lag_req_trees200.pkl")
        p1 = str(BUNDLES / "olmoe_dev_lag_tok_trees200.pkl")
        L, E, k = trace.num_layers, trace.num_experts, trace.top_k
        sim = SplitPhase(load_bundle(p2), 32, "lag_req", "none", diagnose=False, mix_layer0=True,
                         lag_bundle=load_bundle(p2), lag_tok_bundle=load_bundle(p1))
        service = SplitPhaseService(p2, p1, 32, workers=1, num_layers=L, num_experts=E, top_k=k)
        try:
            live = SplitPhasePolicy(service, L, E, 32, k)
            layers = tuple(map(int, trace.metadata["layer_indices"]))
            l2o = {layer: i for i, layer in enumerate(layers)}
            requests = trace.requested_expert_ids.astype(np.int64, copy=False)
            gates = trace.router_weights.astype(np.float64, copy=False)
            for count, (_sequence, view) in enumerate(deploy.iter_slices(trace)):
                if count >= 2:
                    break
                sim.begin_sequence()
                live.begin_sequence()
                for base in range(view.start, view.stop, L):
                    token = int(trace.token_id[base])
                    live.begin_token(token)
                    for index in range(base, base + L):
                        layer = l2o[int(trace.layer_index[index])]
                        sim.step(layer, token, requests[index], gates[index])
                        live.step(layer, token, requests[index], gates[index])
                        self.assertTrue(np.array_equal(sim.resident[layer], live.resident[layer]),
                                        f"resident sets diverge at event {index}")
            self.assertEqual(sim.transfers, live.transfers)
            self.assertGreater(live.eviction_decisions, 0)
        finally:
            service.close()


@unittest.skipUnless(HAVE_TORCH, "torch/transformers missing")
class InterleavedContextTests(unittest.TestCase):
    """Two policies interleaved in one runtime behave exactly as each run alone."""

    def _model(self):
        from transformers import OlmoeConfig, OlmoeForCausalLM

        config = OlmoeConfig(hidden_size=32, intermediate_size=16, num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, vocab_size=64, max_position_embeddings=64,
                             num_experts=8, num_experts_per_tok=2)
        torch.manual_seed(0)
        return OlmoeForCausalLM(config).eval().float()

    def _decode(self, model, runtime, prompt, steps):
        runtime.mode = "prefill"
        with torch.inference_mode():
            out = model(input_ids=prompt, use_cache=True, return_dict=True)
            past = out.past_key_values
            runtime.mode = "decode"
            logits = []
            for token in steps:
                runtime.current_token = int(token)
                runtime.reset_token_counters()
                runtime.routing = []
                out = model(input_ids=torch.tensor([[token]]), past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                logits.append(out.logits[0, -1].clone())
        return torch.stack(logits)

    def test_interleaved_contexts_match_standalone_runs(self) -> None:
        from interleaved_runtime import SLOT_ATTRS, Context, new_slots
        from offload_runtime import OffloadRuntime
        from policy import SimplePolicy

        model = self._model()
        spec = {"architecture": {"layer_indices": [0, 1], "num_moe_layers": 2, "num_experts": 8, "top_k": 2,
                                 "expert_bytes": 3 * 32 * 16 * 4}}
        device = torch.device("cpu")
        torch.manual_seed(2)
        streams = [(torch.randint(0, 64, (1, 5)), torch.randint(0, 64, (10,)).tolist()) for _ in range(4)]

        standalone = {}
        for name in ("lru", "lfu_decay"):
            policy = SimplePolicy(name, 2, 8, 4, 2)
            runtime = OffloadRuntime(model, spec, device, policy, 4)
            outs = [self._decode(model, runtime, p, s) for p, s in streams]
            runtime.release()
            standalone[name] = (policy.transfers, outs)

        a, b = SimplePolicy("lru", 2, 8, 4, 2), SimplePolicy("lfu_decay", 2, 8, 4, 2)
        runtime = OffloadRuntime(model, spec, device, a, 4)
        ctx_a = Context("lru", a, [{x: getattr(layer, x) for x in SLOT_ATTRS} for layer in runtime.layers])
        ctx_b = Context("lfu_decay", b, [new_slots(layer, 4, device) for layer in runtime.layers])
        outs = {"lru": [], "lfu_decay": []}
        for i, (prompt, steps) in enumerate(streams):
            for ctx in ((ctx_a, ctx_b) if i % 2 == 0 else (ctx_b, ctx_a)):
                ctx.use(runtime)
                outs[ctx.name].append(self._decode(model, runtime, prompt, steps))
                ctx.save(runtime)
        runtime.release()
        for name, policy in (("lru", a), ("lfu_decay", b)):
            self.assertEqual(policy.transfers, standalone[name][0], name)
            for got, want in zip(outs[name], standalone[name][1]):
                self.assertTrue(torch.allclose(got, want, atol=1e-5, rtol=1e-5), name)
        self.assertNotEqual(a.transfers, b.transfers)


def _opt_paths() -> None:
    for relative in ("stage3_residency/src", "src"):
        value = str(OPT / relative)
        if value not in sys.path:
            sys.path.insert(0, value)


@unittest.skipUnless(HAVE_TORCH, "torch/transformers missing")
class HostExpertsTests(unittest.TestCase):
    """Experts pinned on the host (Qwen3-30B-A3B and Mixtral-8x7B do not fit the GPU): the
    staged prefill and the slot decode reproduce the resident model on all three architectures,
    with the non-expert weights on the accelerator (CUDA or MPS when available) and the experts on
    the CPU; and the trace generator's own routing hooks (``measurement/``) record the offloaded decode."""

    KINDS = ("olmoe", "qwen3", "mixtral")

    def _model(self, kind: str):
        import transformers as T

        common = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
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
        with torch.no_grad():  # routers may start at zero; give every matrix real weights
            for param in model.parameters():
                if param.dim() >= 2:
                    param.normal_(0.0, 0.3)
        return model

    def _spec(self):
        return {"architecture": {"layer_indices": [0, 1], "num_moe_layers": 2, "num_experts": 8, "top_k": 2,
                                 "expert_bytes": 3 * 32 * 16 * 4}}

    def _device(self):
        if torch.cuda.is_available():  # on a GPU host: pinned host memory and CUDA copies
            return torch.device("cuda")
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

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

    def test_host_experts_equal_resident_on_all_architectures(self) -> None:
        import copy

        from offload_runtime import OffloadRuntime, place_non_experts
        from policy import SimplePolicy

        device = self._device()
        torch.manual_seed(1)
        prompt = torch.randint(0, 64, (1, 7))
        steps = torch.randint(0, 64, (14,)).tolist()
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                base = self._model(kind)
                resident_model = copy.deepcopy(base).to(device)
                resident = OffloadRuntime(resident_model, self._spec(), device, None, 8)
                reference, routing = self._decode(resident_model, resident, device, prompt, steps)
                resident.release()

                hosted = copy.deepcopy(base)
                counts = place_non_experts(hosted, device, [0, 1])
                self.assertEqual(counts["host_bytes"], 2 * 8 * 3 * 32 * 16 * 4)
                for name, param in hosted.named_parameters():
                    on_host = ".mlp.experts." in name
                    self.assertEqual(param.device.type, "cpu" if on_host else device.type, name)
                policy = SimplePolicy("lru", 2, 8, 4, 2)
                runtime = OffloadRuntime(hosted, self._spec(), device, policy, 4, experts_on_host=True)
                self.assertTrue(all(layer.on_host for layer in runtime.layers))
                measured, routing_hosted = self._decode(hosted, runtime, device, prompt, steps)
                runtime.release()
                self.assertTrue(torch.allclose(reference, measured, atol=1e-4, rtol=1e-4),
                                f"{kind}: max diff {(reference - measured).abs().max().item():.2e}")
                self.assertEqual(routing, routing_hosted, kind)
                self.assertGreater(policy.transfers, 0)

    def test_resident_reference_refused_with_host_experts(self) -> None:
        from offload_runtime import OffloadRuntime, place_non_experts

        model = self._model("qwen3")
        place_non_experts(model, torch.device("cpu"), [0, 1])
        with self.assertRaises(RuntimeError):
            OffloadRuntime(model, self._spec(), torch.device("cpu"), None, 8, experts_on_host=True)

    @unittest.skipUnless((OPT / "stage3_residency").exists(), "needs the measurement code")
    def test_trace_generator_hooks_record_the_offloaded_decode(self) -> None:
        """``fresh_trace_offloaded.py`` runs the trace generator's own decode loop and routing
        hooks on the offloaded model ('auto' mode: a multi-token call is prefill)."""

        _opt_paths()
        from expert_analysis.modeling import discover_moe_layers
        from residency_headroom.trace_generation import DecodeRoutingCapture

        from offload_runtime import OffloadRuntime, place_non_experts
        from policy import SimplePolicy

        device = self._device()
        torch.manual_seed(2)
        prompt = torch.randint(0, 64, (1, 6)).to(device)
        for kind in ("qwen3", "mixtral"):
            with self.subTest(kind=kind):
                model = self._model(kind)
                place_non_experts(model, device, [0, 1])
                runtime = OffloadRuntime(model, self._spec(), device, SimplePolicy("lfu_decay", 2, 8, 4, 2), 4,
                                         experts_on_host=True)
                runtime.mode = "auto"
                specs = discover_moe_layers(model)
                self.assertEqual(len(specs), 2)
                with torch.inference_mode(), DecodeRoutingCapture(specs) as capture:
                    out = model(input_ids=prompt, use_cache=True, return_dict=True)  # prefill: capture inactive
                    past = out.past_key_values
                    for _ in range(10):
                        token = torch.argmax(out.logits[:, -1, :].float(), dim=-1)
                        runtime.routing = []
                        capture.begin_token()
                        out = model(input_ids=token[:, None], past_key_values=past, use_cache=True, return_dict=True)
                        records = capture.finish_token()
                        past = out.past_key_values
                        self.assertEqual([sorted(r.expert_ids.tolist()) for r in records],
                                         [sorted(req.tolist()) for _o, req, _g in runtime.routing])
                runtime.release()



@unittest.skipUnless(HAVE_TORCH, "torch/transformers missing")
class SharedSlotInterleavingTests(unittest.TestCase):
    """Three policies rotating on one shared, refilled slot set (``interleaved_multi.py``) behave
    exactly as each policy run alone, with the experts on the host."""

    def test_rotating_policies_on_shared_slots_match_standalone_runs(self) -> None:
        import copy

        from interleaved_multi import SharedSlotContext, rotation, run_interleaved
        from offload_runtime import OffloadRuntime, place_non_experts
        from policy import SimplePolicy

        helper = HostExpertsTests()
        device = helper._device()
        base = helper._model("qwen3")
        spec = helper._spec()
        torch.manual_seed(3)
        streams = [(torch.randint(0, 64, (1, 5)), torch.randint(0, 64, (9,)).tolist()) for _ in range(5)]
        names = ["lru", "lfu_decay", "lfu"]
        self.assertEqual(rotation(names, 1), ["lfu_decay", "lfu", "lru"])

        standalone = {}
        for name in names:
            model = copy.deepcopy(base)
            place_non_experts(model, device, [0, 1])
            policy = SimplePolicy(name, 2, 8, 4, 2)
            runtime = OffloadRuntime(model, spec, device, policy, 4, experts_on_host=True)
            outs = [helper._decode(model, runtime, device, p, s)[0] for p, s in streams]
            runtime.release()
            standalone[name] = (policy.transfers, outs)

        model = copy.deepcopy(base)
        place_non_experts(model, device, [0, 1])
        policies = {name: SimplePolicy(name, 2, 8, 4, 2) for name in names}
        runtime = OffloadRuntime(model, spec, device, policies["lru"], 4, experts_on_host=True)
        contexts = [SharedSlotContext(name, policies[name], runtime) for name in names]
        outs = {name: [] for name in names}

        def decode(item):
            prompt, steps = streams[item["i"]]
            logits, _routing = helper._decode(model, runtime, device, prompt, steps)
            outs[runtime_policy_name()].append(logits)
            return {"wall_ms": 1.0, "tokens": 1, "expert_transfers": 0}

        def runtime_policy_name():
            return next(n for n, p in policies.items() if p is runtime.policy)

        order, _ = run_interleaved(runtime, contexts, [{"i": i} for i in range(len(streams))], decode, log=lambda m: None)
        runtime.release()
        self.assertEqual(order[0], names)
        self.assertEqual(order[2], ["lfu", "lru", "lfu_decay"])
        for name in names:
            self.assertEqual(policies[name].transfers, standalone[name][0], name)
            for got, want in zip(outs[name], standalone[name][1]):
                self.assertTrue(torch.allclose(got, want, atol=1e-4, rtol=1e-4), name)
            self.assertEqual(next(c for c in contexts if c.name == name).refills, len(streams))
        self.assertGreater(len({policies[n].transfers for n in names}), 1)


@unittest.skipUnless(HAVE_TORCH, "torch/transformers missing")
class PrefetchRuntimeTests(unittest.TestCase):
    """Next-layer prefetching (T4) changes no hidden state, and the runtime's demand and prefetch
    copies equal a replay of the same policy driven by the routing and guesses it observed."""

    def _model(self, layers: int = 3):
        from transformers import OlmoeConfig, OlmoeForCausalLM

        config = OlmoeConfig(hidden_size=32, intermediate_size=16, num_hidden_layers=layers, num_attention_heads=4,
                             num_key_value_heads=2, vocab_size=64, max_position_embeddings=64,
                             num_experts=8, num_experts_per_tok=2)
        torch.manual_seed(0)
        return OlmoeForCausalLM(config).eval().float()

    def _decode(self, model, runtime, prompt, steps):
        runtime.mode = "prefill"
        tokens = []
        with torch.inference_mode():
            out = model(input_ids=prompt, use_cache=True, return_dict=True)
            past = out.past_key_values
            runtime.mode = "decode"
            logits = []
            for token in steps:
                runtime.current_token = int(token)
                runtime.reset_token_counters()
                runtime.routing = []
                runtime.guesses = []
                out = model(input_ids=torch.tensor([[token]]), past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                logits.append(out.logits[0, -1].clone())
                tokens.append((list(runtime.routing), list(runtime.guesses)))
        return torch.stack(logits), tokens

    def test_prefetch_changes_no_hidden_state_and_replays_exactly(self) -> None:
        from offload_runtime import OffloadRuntime, PrefetchConfig
        from policy import SimplePolicy

        L = 3
        model = self._model(L)
        spec = {"architecture": {"layer_indices": list(range(L)), "num_moe_layers": L, "num_experts": 8, "top_k": 2,
                                 "expert_bytes": 3 * 32 * 16 * 4}}
        device = torch.device("cpu")
        torch.manual_seed(5)
        prompt = torch.randint(0, 64, (1, 6))
        steps = torch.randint(0, 64, (16,)).tolist()

        resident = OffloadRuntime(model, spec, device, None, 8)
        reference, _ = self._decode(model, resident, prompt, steps)
        resident.release()

        for variant in ("g1", "g1n"):
            pf = PrefetchConfig(m=3, q=2, variant=variant)
            policy = SimplePolicy("lfu_decay", L, 8, 4, 2)
            runtime = OffloadRuntime(model, spec, device, policy, 4, prefetch=pf)
            measured, tokens = self._decode(model, runtime, prompt, steps)
            runtime.release()
            self.assertTrue(torch.allclose(reference, measured, atol=1e-5, rtol=1e-5), variant)
            self.assertGreater(policy.prefetch_transfers, 0, variant)

            replay = SimplePolicy("lfu_decay", L, 8, 4, 2)
            for routing, guesses in tokens:
                by_layer = {ordinal: guess for ordinal, guess in guesses}
                self.assertEqual(sorted(by_layer), list(range(1, L)))
                for ordinal, request, gates in routing:
                    replay.step(ordinal, 0, request, gates)
                    if ordinal + 1 < L:
                        replay.prefetch(ordinal + 1, by_layer[ordinal + 1], pf.q)
            self.assertEqual(policy.transfers, replay.transfers, variant)
            self.assertEqual(policy.prefetch_transfers, replay.prefetch_transfers, variant)
            self.assertEqual(policy.prefetch_hits, replay.prefetch_hits, variant)
            # the slots hold exactly the resident experts, with the original weights
            for layer in runtime.layers:
                ids = policy.resident_experts(layer.ordinal)
                self.assertEqual(sorted(int(e) for e in np.flatnonzero(layer.slot_of >= 0)), sorted(ids.tolist()))
                for expert in ids:
                    slot = int(layer.slot_of[expert])
                    self.assertTrue(torch.equal(layer.slot_gate_up[slot], layer.resident_gate_up[int(expert)]))

    def test_guess_equals_the_recorders_ranking(self) -> None:
        from offload_runtime import OffloadRuntime, PrefetchConfig
        from policy import SimplePolicy
        from record_guesses import guess_topm

        L = 3
        model = self._model(L)
        spec = {"architecture": {"layer_indices": list(range(L)), "num_moe_layers": L, "num_experts": 8, "top_k": 2,
                                 "expert_bytes": 3 * 32 * 16 * 4}}
        runtime = OffloadRuntime(model, spec, torch.device("cpu"), SimplePolicy("lfu_decay", L, 8, 4, 2), 4,
                                 prefetch=PrefetchConfig(m=4, q=1))
        hidden = torch.randn(1, 32)
        layer = runtime.layers[0]
        got = layer._guess(hidden, runtime.layers[1], runtime.prefetch).tolist()
        want = guess_topm(torch, runtime.layers[1].block, hidden, 4).tolist()
        runtime.release()
        self.assertEqual(got, want)

    def test_bf16_logits_read_on_the_host_rank_like_the_device(self) -> None:
        """The CUDA path copies bf16 logits as int16 and widens them on the host; the ranking must equal
        the ranking of the same logits widened by torch (exact: bf16 is the top half of float32)."""

        torch.manual_seed(7)
        logits = (torch.randn(64) * 3).to(torch.bfloat16)
        bits = logits.view(torch.int16).numpy().view(np.uint16).astype(np.uint32) << 16
        widened = bits.view(np.float32)
        self.assertTrue(np.array_equal(widened, logits.float().numpy()))
        self.assertEqual(np.argsort(-widened, kind="stable")[:12].tolist(),
                         np.argsort(-logits.float().numpy(), kind="stable")[:12].tolist())


@unittest.skipUnless(HAVE_SKLEARN and HAVE_NUMBA and (OPT / "stage3_residency").exists()
                     and (BUNDLES / "olmoe_dev_lag_req_trees200.pkl").exists(),
                     "needs the measurement data and the development split-phase rankers")
class SplitPhasePrefetchTests(unittest.TestCase):
    """With prefetching, the running service (victims from the published ranking, unrequested
    prefetches ranked last) makes the simulator's decisions exactly."""

    def test_service_prefetch_matches_simulation(self) -> None:
        sys.path.insert(0, str(ROOT / "analysis"))
        from splitphase import SplitPhase, load_bundle, load_dev
        from splitphase_service import SplitPhasePolicy, SplitPhaseService

        inputs, _fit, deploy = load_dev()
        trace = inputs.trace
        p2 = str(BUNDLES / "olmoe_dev_lag_req_trees200.pkl")
        p1 = str(BUNDLES / "olmoe_dev_lag_tok_trees200.pkl")
        L, E, k = trace.num_layers, trace.num_experts, trace.top_k
        sim = SplitPhase(load_bundle(p2), 32, "lag_req", "none", diagnose=False, mix_layer0=True,
                         lag_bundle=load_bundle(p2), lag_tok_bundle=load_bundle(p1))
        service = SplitPhaseService(p2, p1, 32, workers=1, num_layers=L, num_experts=E, top_k=k)
        rng = np.random.default_rng(0)
        try:
            live = SplitPhasePolicy(service, L, E, 32, k)
            layers = tuple(map(int, trace.metadata["layer_indices"]))
            l2o = {layer: i for i, layer in enumerate(layers)}
            requests = trace.requested_expert_ids.astype(np.int64, copy=False)
            gates = trace.router_weights.astype(np.float64, copy=False)
            for count, (_sequence, view) in enumerate(deploy.iter_slices(trace)):
                if count >= 2:
                    break
                sim.begin_sequence()
                live.begin_sequence()
                for base in range(view.start, view.stop, L):
                    token = int(trace.token_id[base])
                    live.begin_token(token)
                    for index in range(base, base + L):
                        layer = l2o[int(trace.layer_index[index])]
                        sim.step(layer, token, requests[index], gates[index])
                        live.step(layer, token, requests[index], gates[index])
                        self.assertTrue(np.array_equal(sim.resident[layer], live.resident[layer]), f"demand {index}")
                        if layer + 1 < L:
                            guess = list(dict.fromkeys(requests[index + 1][:4].tolist()
                                                       + rng.choice(E, 6, replace=False).tolist()))[:8]
                            a = sim.prefetch(layer + 1, token, np.asarray(guess), 3)
                            b = live.prefetch(layer + 1, np.asarray(guess), 3)
                            self.assertEqual(sorted(a[0].tolist()), sorted(b[0].tolist()), f"victims {index}")
                            self.assertEqual(sorted(a[1].tolist()), sorted(b[1].tolist()), f"loads {index}")
            self.assertEqual(sim.transfers, live.transfers)
            self.assertEqual(sim.prefetch_transfers, live.prefetch_transfers)
            self.assertEqual(sim.prefetch_hits, live.prefetch_hits)
            self.assertGreater(live.prefetch_transfers, 0)
        finally:
            service.close()


if __name__ == "__main__":
    unittest.main()
