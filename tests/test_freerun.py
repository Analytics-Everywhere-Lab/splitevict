"""T7 checks that need no GPU: the free-running runner and the policy selector.

* ``interleaved_freerun.py`` hands ``teacher_force=False`` to the runtime whatever ``interleaved_h100.py`` asks for,
  restores ``interleaved_h100.run_path`` afterwards, and marks its output.
* ``run_path`` without teacher forcing reproduces a trace that is the model's own greedy decode (tokens, routing and
  copies as with teacher forcing), and feeds back its own token where the recorded one differs.
* ``selector.py``: at the calibrated copy cost the prediction is the calibration run's measured difference; the
  decision rule and the evaluation's clear-loss / clear-gain classes.
"""

from __future__ import annotations

import json
import sys
import tempfile
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


class FreeRunningWrapperTests(unittest.TestCase):
    def test_wrapper_forces_free_running_and_restores_the_runner(self) -> None:
        import interleaved_freerun
        import interleaved_h100

        seen = []
        sealed_main, sealed_inner = interleaved_h100.main, interleaved_freerun._teacher_forced_run_path
        sealed_run_path = interleaved_h100.run_path

        spec = {"trace_generation": {"stop_on_eos": True, "max_prompt_tokens": 512}}
        written = []

        def fake_main(argv):
            out = interleaved_h100.run_path("runtime", "tokenizer", spec, {}, teacher_force=True, limit_sequences=None,
                                            log=None)
            written.append(argv[argv.index("--output") + 1])
            Path(written[-1]).write_text(json.dumps({"arms": {}, "seen": out}))

        def recorder(runtime, tokenizer, spec_seen, workload, *, teacher_force, **kwargs):
            seen.append((teacher_force, spec_seen["trace_generation"]["stop_on_eos"]))
            return "decoded"

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run.json"
            interleaved_h100.main, interleaved_freerun._teacher_forced_run_path = fake_main, recorder
            try:
                interleaved_freerun.main(["--spec", "x", "--output", str(output)])
            finally:
                interleaved_h100.main, interleaved_freerun._teacher_forced_run_path = sealed_main, sealed_inner
            out = json.loads(output.read_text())
            self.assertEqual(written, [str(output) + ".partial"])  # the runner never writes the final name
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["run.json"])
        self.assertEqual(seen, [(False, False)])
        self.assertTrue(spec["trace_generation"]["stop_on_eos"])  # the caller's spec is not modified
        self.assertEqual(out["decoding"], "free_running")
        self.assertEqual(out["runner"], "interleaved_freerun.py")
        self.assertEqual(out["seen"], "decoded")
        self.assertIs(interleaved_h100.run_path, sealed_run_path)
        self.assertIs(sealed_run_path, sealed_inner)


@unittest.skipUnless(HAVE_TORCH, "torch/transformers missing")
class FreeRunningRunPathTests(unittest.TestCase):
    STEPS = 12

    def _model(self):
        from transformers import OlmoeConfig, OlmoeForCausalLM

        config = OlmoeConfig(hidden_size=32, intermediate_size=16, num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, vocab_size=64, max_position_embeddings=64,
                             num_experts=8, num_experts_per_tok=2)
        torch.manual_seed(0)
        return OlmoeForCausalLM(config).eval().float()

    def _greedy_trace(self, model, arch, prompt):
        """The model's own greedy decode through a resident runtime: tokens and each step's requests per layer."""

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

    def _run(self, model, arch, prompt, tokens, requests, teacher_force, eos=None, run_path=None):
        from offload_runtime import OffloadRuntime
        from policy import SimplePolicy

        if run_path is None:
            from offload_runtime import run_path

        class Tokenizer:
            eos_token_id = eos

            def __call__(self, text, **kwargs):
                return {"input_ids": prompt.clone(), "attention_mask": torch.ones_like(prompt)}

        spec = {"architecture": arch, "trace_generation": {"max_prompt_tokens": 512, "stop_on_eos": True},
                "sequences": [{"sequence_id": 0, "prompt": "p", "prompt_length": int(prompt.shape[1]),
                               "generated_token_ids": list(tokens),
                               "trace_requested_expert_ids": np.asarray(requests).reshape(-1).tolist()}]}
        workload = {"name": "test", "regime": "test",
                    "sequences": [{"position": 0, "source_sequence_id": 0, "domain": "general"}]}
        policy = SimplePolicy("lfu_decay", 2, 8, 4, 2)
        runtime = OffloadRuntime(model, spec, torch.device("cpu"), policy, 4)
        try:
            result = run_path(runtime, Tokenizer(), spec, workload, teacher_force=teacher_force, limit_sequences=None,
                              log=lambda message: None)
        finally:
            runtime.release()
        return result["sequences"][0]

    def test_free_running_reproduces_its_own_greedy_trace(self) -> None:
        model = self._model()
        arch = {"layer_indices": [0, 1], "num_moe_layers": 2, "num_experts": 8, "top_k": 2, "expert_bytes": 3 * 32 * 16 * 4}
        torch.manual_seed(3)
        prompt = torch.randint(0, 64, (1, 6))
        tokens, requests = self._greedy_trace(model, arch, prompt)
        forced = self._run(model, arch, prompt, tokens, requests, teacher_force=True)
        free = self._run(model, arch, prompt, tokens, requests, teacher_force=False)
        for row in (forced, free):
            self.assertEqual(row["tokens"], self.STEPS)
            self.assertEqual(row["routing_matches"], row["routing_events"])
            self.assertEqual(row["token_agreements"], row["token_comparisons"])
            self.assertTrue(row["first_token_agrees"])
        self.assertEqual(free["expert_transfers"], forced["expert_transfers"])
        self.assertGreater(free["expert_transfers"], 0)

        # a recorded token the model would not emit: free-running feeds back its own token, so routing still matches the
        # greedy decode at every step and only that one token comparison fails
        changed = list(tokens)
        changed[5] = (changed[5] + 1) % 64
        free = self._run(model, arch, prompt, changed, requests, teacher_force=False)
        self.assertEqual(free["routing_matches"], free["routing_events"])
        self.assertEqual(free["token_agreements"], free["token_comparisons"] - 1)
        forced = self._run(model, arch, prompt, changed, requests, teacher_force=True)  # the changed token is fed in
        self.assertLess(forced["routing_matches"], forced["routing_events"])

    def test_a_trace_ending_at_eos_is_reproduced_with_its_eos_step(self) -> None:
        """The trace generator feeds EOS through the model before it stops; so must the free-running runner."""

        from interleaved_freerun import free_running_run_path

        model = self._model()
        arch = {"layer_indices": [0, 1], "num_moe_layers": 2, "num_experts": 8, "top_k": 2, "expert_bytes": 3 * 32 * 16 * 4}
        torch.manual_seed(3)
        prompt = torch.randint(0, 64, (1, 6))
        tokens, requests = self._greedy_trace(model, arch, prompt)
        eos = tokens[-1]
        self.assertNotIn(eos, tokens[:-1])  # the greedy decode emits this token only at its last step: call it EOS
        forced = self._run(model, arch, prompt, tokens, requests, teacher_force=True, eos=eos)
        free = self._run(model, arch, prompt, tokens, requests, teacher_force=True, eos=eos, run_path=free_running_run_path)
        for row in (forced, free):
            self.assertEqual(row["tokens"], self.STEPS)
            self.assertEqual(row["routing_matches"], row["routing_events"])
            self.assertEqual(row["token_agreements"], row["token_comparisons"])
        self.assertEqual(free["expert_transfers"], forced["expert_transfers"])
        # run_path alone stops before feeding EOS: one step short
        bare = self._run(model, arch, prompt, tokens, requests, teacher_force=False, eos=eos)
        self.assertEqual(bare["tokens"], self.STEPS - 1)


class SelectorTests(unittest.TestCase):
    def _entry(self, wall_s=95.0, copies_s=8.0):
        from dev_costs import costs

        def arm(wall, copies, policy):
            return {"totals": {"tokens": 100, "expert_transfers": copies * 100, "copy_ms": copies * 100 * 0.5,
                               "issue_ms": copies * 100 * 0.01, "ms_per_token_policy": policy},
                    "sequences": [{"source_sequence_id": 0, "wall_ms": wall * 100, "tokens": 100}]}

        return costs({"passes": 1, "arms": {"lfu_decay": arm(100.0, 10.0, 1.0), "splitphase": arm(wall_s, copies_s, 1.5)}})

    def test_calibrated_prediction_is_the_measured_difference(self) -> None:
        from selector import decide, predicted_gain

        entry = self._entry()
        self.assertAlmostEqual(predicted_gain(entry, 1), 0.05, places=12)
        # at twice the copy cost, two copies fewer per token save 2 x 0.51 ms more of a token that grows by 10 x 0.51 ms
        self.assertAlmostEqual(predicted_gain(entry, 2), (5.0 + 2 * 0.51) / (100.0 + 10 * 0.51), places=12)
        loss = self._entry(wall_s=101.0)
        self.assertEqual(decide(loss, [1])["1"]["policy"], "lfu_decay")
        self.assertEqual(decide(entry, [1])["1"]["policy"], "splitphase")
        near = self._entry(wall_s=99.2)  # +0.8%: below the one-point margin
        self.assertEqual(decide(near, [1])["1"]["policy"], "lfu_decay")

    def test_evaluation_classes_and_regret(self) -> None:
        from selector import evaluate

        result = evaluate({"a": "lfu_decay", "b": "splitphase", "c": "splitphase", "d": "lfu_decay"},
                          {"a": {"gain": -0.01, "ci": [-0.02, -0.005]}, "b": {"gain": 0.05, "ci": [0.03, 0.07]},
                           "c": {"gain": 0.005, "ci": [-0.001, 0.011]}, "d": {"gain": 0.03, "ci": [0.02, 0.04]}})
        self.assertTrue(result["avoids_every_clear_loss"])
        self.assertFalse(result["keeps_every_clear_gain"])  # d is a clear gain left off
        self.assertEqual(result["settings"]["c"]["kind"], "no_claim")
        self.assertAlmostEqual(result["regret_sum"]["selector"], 0.03)
        self.assertAlmostEqual(result["regret_sum"]["always_splitphase"], 0.01)
        self.assertAlmostEqual(result["regret_sum"]["always_lfu"], 0.085)


if __name__ == "__main__":
    unittest.main()
