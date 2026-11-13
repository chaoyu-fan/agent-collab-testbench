import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from weightlab import prepare, load_model, demo_patch, evaluate, read_patch


class TestBlindBenchmark(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        prepare(self.root / "run", seed=123)
        self.participant = self.root / "run" / "participant"
        self.organizer = self.root / "run" / "organizer"

    def test_isolation(self):
        self.assertFalse((self.participant / "teacher.npz").exists())
        self.assertFalse((self.participant / "hidden_test.npz").exists())
        with (self.participant / "rules.json").open() as f:
            public = json.load(f)
        self.assertNotIn("baseline_test_accuracy_organizer_only", public)
        self.assertNotIn("seed", public)

    def test_zero_patch_no_gain(self):
        patch = self.root / "zero.npz"
        demo_patch(self.participant / "model.npz", patch)
        result = evaluate(self.participant, self.organizer, patch)
        self.assertEqual(result["accuracy_gain_pp"], 0)
        self.assertEqual(result["changed_entries"], 0)

    def test_oversized_patch_rejected(self):
        model = load_model(self.participant / "model.npz")
        patch = self.root / "bad.npz"
        np.savez_compressed(patch, delta_w2=np.ones_like(model["w2"]))
        with self.assertRaises(ValueError):
            read_patch(patch, model["w2"])

    def test_nonfinite_patch_rejected(self):
        model = load_model(self.participant / "model.npz")
        patch = self.root / "nan.npz"
        delta = np.zeros_like(model["w2"])
        delta[0, 0] = np.nan
        np.savez_compressed(patch, delta_w2=delta)
        with self.assertRaises(ValueError):
            read_patch(patch, model["w2"])

    def test_manifest_detects_tamper(self):
        patch = self.root / "zero.npz"
        demo_patch(self.participant / "model.npz", patch)
        with np.load(self.participant / "model.npz") as data:
            mod = {k: data[k].copy() for k in data.files}
        mod["w2"][0, 0] += 1
        np.savez_compressed(self.participant / "model.npz", **mod)
        with self.assertRaises(ValueError):
            evaluate(self.participant, self.organizer, patch)

    def test_dev_tamper_rejected(self):
        patch = self.root / "zero.npz"
        demo_patch(self.participant / "model.npz", patch)
        with (self.participant / "dev.jsonl").open("a", encoding="utf-8") as f:
            f.write("\n")
        with self.assertRaises(ValueError):
            evaluate(self.participant, self.organizer, patch)

    def test_reproducibility(self):
        prepare(self.root / "again", seed=123)
        a = load_model(self.participant / "model.npz")
        b = load_model(self.root / "again" / "participant" / "model.npz")
        self.assertTrue(all(np.array_equal(a[k], b[k]) for k in a))
        self.assertEqual(
            (self.participant / "dev.jsonl").read_bytes(),
            (self.root / "again" / "participant" / "dev.jsonl").read_bytes(),
        )


if __name__ == "__main__":
    unittest.main()
