"""T8 checks that need no GPU: experts staged on a drive and read back for every demand copy.

* The staged files hold every expert's bytes at its offset (read back with the reader the runs use).
* Decoding with every demand copy read from the files gives the same logits, routing and transfer count as decoding
  with copies from host memory, and makes exactly one read per transfer.
* The runner refuses a copy repeat other than 1.
"""

from __future__ import annotations

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


@unittest.skipUnless(HAVE_TORCH, "torch/transformers missing")
class NvmeExpertsTests(unittest.TestCase):
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

    def _decode(self, model, runtime, prompt, steps):
        runtime.mode = "prefill"
        with torch.inference_mode():
            out = model(input_ids=prompt, use_cache=True, return_dict=True)
            past = out.past_key_values
            logits, routing = [], []
            runtime.mode = "decode"
            for token in steps:
                runtime.current_token = int(token)
                runtime.reset_token_counters()
                runtime.routing = []
                out = model(input_ids=torch.tensor([[token]]), past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                logits.append(out.logits[0, -1].clone())
                routing.append([(o, r.tolist()) for o, r, _g in runtime.routing])
        return torch.stack(logits), routing

    def test_drive_copies_equal_host_copies(self) -> None:
        import interleaved_nvme
        from offload_runtime import OffloadedMoeLayer, OffloadRuntime
        from policy import SimplePolicy

        model = self._tiny_model()
        device = torch.device("cpu")
        torch.manual_seed(1)
        prompt = torch.randint(0, 64, (1, 6))
        steps = torch.randint(0, 64, (16,)).tolist()

        host_policy = SimplePolicy("lru", 2, 8, 4, 2)
        host = OffloadRuntime(model, self._spec(), device, host_policy, 4)
        reference, routing = self._decode(model, host, prompt, steps)
        host.release()

        drive_policy = SimplePolicy("lru", 2, 8, 4, 2)
        drive = OffloadRuntime(model, self._spec(), device, drive_policy, 4)
        with tempfile.TemporaryDirectory() as tmp:
            store = interleaved_nvme.NvmeExperts(drive, Path(tmp) / "experts", buffers=2)
            drive.nvme = store
            self.assertTrue(store.readback_equal)
            self.assertEqual(store.stride % interleaved_nvme.ALIGN, 0)
            self.assertEqual([p.stat().st_size for p in store.paths], [8 * store.stride] * 2)
            for layer in drive.layers:  # every expert at its offset
                gate_up, down = store._host_bytes(layer)
                for expert in range(layer.E):
                    index, _ = store.read(layer.ordinal, expert)
                    got = np.frombuffer(store.views[index], dtype=np.uint8)
                    self.assertTrue(np.array_equal(got[: store.gate_up_bytes], gate_up[expert]))
                    self.assertTrue(np.array_equal(got[store.gate_up_bytes: store.expert_bytes], down[expert]))
                    self.assertFalse(got[store.expert_bytes:].any())
            sealed_copy = OffloadedMoeLayer.copy_expert
            OffloadedMoeLayer.copy_expert = interleaved_nvme.nvme_copy_expert
            try:
                reads_before = store.reads
                measured, routing_drive = self._decode(model, drive, prompt, steps)
                reads = store.reads - reads_before
            finally:
                OffloadedMoeLayer.copy_expert = sealed_copy
            probe = store.probe(reads=8)
            for fd in store.fds:
                import os

                os.close(fd)
        drive.release()
        self.assertTrue(torch.equal(reference, measured), "copies read from the drive changed the logits")
        self.assertEqual(routing, routing_drive)
        self.assertEqual(host_policy.transfers, drive_policy.transfers)
        self.assertGreater(drive_policy.transfers, 0)
        self.assertEqual(reads, drive_policy.transfers)
        self.assertEqual(probe["reads"], 8)
        self.assertGreater(probe["ms_mean"], 0.0)
        self.assertIs(OffloadedMoeLayer.copy_expert, sealed_copy)

    def test_runner_refuses_a_copy_repeat(self) -> None:
        import interleaved_nvme

        with self.assertRaises(SystemExit):
            interleaved_nvme.main(["--nvme-dir", "/nonexistent", "--copy-repeat", "2", "--output", "x.json"])


if __name__ == "__main__":
    unittest.main()
