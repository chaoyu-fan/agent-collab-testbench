"""A reproducible, CPU-only blind weight-repair benchmark (NumPy).

This is a *micro-benchmark*, not a substitute for genuine LLM evaluation.
The organizer's secret must live outside the contestant's filesystem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from pathlib import Path
from typing import Any

import numpy as np

PARAMS = ("w1", "b1", "w2", "b2")
MAX_CHANGED = 24
MAX_NORM = 8.0

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    h = np.tanh(x @ model["w1"] + model["b1"])
    return np.argmax(h @ model["w2"] + model["b2"], axis=1)


def accuracy(model: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(predict(model, x) == y))


def load_model(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != set(PARAMS):
            raise ValueError(f"model must have exactly {PARAMS}")
        return {k: z[k].astype(np.float64) for k in PARAMS}


def generate(seed: int = 20261113) -> tuple[dict, dict, tuple, tuple]:
    rng = np.random.default_rng(seed)
    d, h, k = 16, 32, 4
    teacher = {
        "w1": rng.normal(0, 0.75, (d, h)),
        "b1": rng.normal(0, 0.35, (h,)),
        "w2": rng.normal(0, 0.9, (h, k)),
        "b2": rng.normal(0, 0.3, (k,)),
    }
    student = {key: value.copy() for key, value in teacher.items()}
    # Perturb a few rows of the hidden-to-logit map. The contestant
    # receives only the resulting weights, not these indices or the teacher.
    rows = rng.choice(h, size=5, replace=False)
    for row in rows:
        student["w2"][row] += rng.normal(0, 3.0, (k,))
    x_dev = rng.normal(0, 1, (400, d))
    x_test = rng.normal(0, 1, (600, d))
    y_dev = predict(teacher, x_dev)
    y_test = predict(teacher, x_test)
    return teacher, student, (x_dev, y_dev), (x_test, y_test)


def prepare(out: Path, seed: int | None) -> dict[str, Any]:
    if seed is None:
        seed = secrets.randbits(63)
    participant = out / "participant"
    organizer = out / "organizer"
    participant.mkdir(parents=True, exist_ok=True)
    organizer.mkdir(parents=True, exist_ok=True)
    teacher, student, (x_dev, y_dev), (x_test, y_test) = generate(seed)
    np.savez_compressed(participant / "model.npz", **student)
    with (participant / "dev.jsonl").open("w", encoding="utf-8") as f:
        for idx, (x, y) in enumerate(zip(x_dev, y_dev)):
            f.write(json.dumps({"id": f"dev-{idx:04d}", "x": x.tolist(), "label": int(y)}) + "\n")
    np.savez_compressed(organizer / "hidden_test.npz", x=x_test, y=y_test)
    # Teacher remains strictly organizer-side for auditing, never used to grade edits.
    np.savez_compressed(organizer / "teacher.npz", **teacher)
    manifest = {
        "schema_version": 1, "seed": seed, "model_type": "tanh_classifier",
        "target_tensor": "w2", "max_changed_entries": MAX_CHANGED,
        "max_frobenius_norm": MAX_NORM, "dev_size": len(y_dev), "test_size": len(y_test),
        "checkpoint_sha256": sha256(participant / "model.npz"),
        "dev_sha256": sha256(participant / "dev.jsonl"),
        "baseline_dev_accuracy": accuracy(student, x_dev, y_dev),
        "baseline_test_accuracy_organizer_only": accuracy(student, x_test, y_test),
    }
    (organizer / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    public = {k: v for k, v in manifest.items() if not k.startswith("baseline_test") and k != "seed"}
    (participant / "rules.json").write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"participant": str(participant), "organizer": str(organizer), "public_rules": public}, indent=2))
    return manifest


def read_patch(path: Path, w2: np.ndarray) -> tuple[np.ndarray, int, float]:
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"delta_w2"}:
            raise ValueError("patch must contain only 'delta_w2'")
        delta = np.asarray(z["delta_w2"], dtype=np.float64)
    if delta.shape != w2.shape or not np.all(np.isfinite(delta)):
        raise ValueError("invalid patch shape or non-finite values")
    changed = int(np.count_nonzero(delta))
    norm = float(np.linalg.norm(delta))
    if changed > MAX_CHANGED or norm > MAX_NORM + 1e-10:
        raise ValueError(f"patch exceeds limits: changed={changed}, norm={norm:.4f}")
    return delta, changed, norm


def evaluate(participant: Path, organizer: Path, patch: Path, output: Path | None = None) -> dict[str, Any]:
    manifest = json.loads((organizer / "manifest.json").read_text(encoding="utf-8"))
    if sha256(participant / "model.npz") != manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from frozen organizer manifest")
    if sha256(participant / "dev.jsonl") != manifest["dev_sha256"]:
        raise ValueError("development set differs from frozen organizer manifest")
    student = load_model(participant / "model.npz")
    delta, changed, norm = read_patch(patch, student["w2"])
    with np.load(organizer / "hidden_test.npz", allow_pickle=False) as z:
        x, y = z["x"], z["y"]
    baseline = accuracy(student, x, y)
    student["w2"] += delta
    improved = accuracy(student, x, y)
    result = {
        "schema_version": 1, "checkpoint_sha256": manifest["checkpoint_sha256"],
        "patch_sha256": sha256(patch),
        "test_n": int(len(y)), "baseline_accuracy": baseline,
        "candidate_accuracy": improved, "accuracy_gain_pp": round((improved-baseline)*100, 4),
        "changed_entries": changed, "patch_frobenius_norm": norm,
        "valid_patch": True,
        "note": "CPU micro-benchmark only. Agent call cost is not independently verified.",
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def demo_patch(model_path: Path, out: Path) -> None:
    model = load_model(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, delta_w2=np.zeros_like(model["w2"]))
    print(f"Wrote no-op patch: {out} (sanity check only, not an improvement)")


def main() -> None:
    p = argparse.ArgumentParser(description="Blind weight-repair benchmark organizer CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare", help="create separated contestant and organizer assets")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--seed", type=int, default=None, help="organizer-only; defaults to a random secret seed")
    b = sub.add_parser("demo-patch", help="create a valid no-op delta for smoke testing")
    b.add_argument("--model", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("evaluate", help="organizer-only hidden test scoring")
    c.add_argument("--participant", type=Path, required=True)
    c.add_argument("--organizer", type=Path, required=True)
    c.add_argument("--patch", type=Path, required=True)
    c.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.cmd == "prepare":
        prepare(args.out, args.seed)
    elif args.cmd == "demo-patch":
        demo_patch(args.model, args.out)
    elif args.cmd == "evaluate":
        evaluate(args.participant, args.organizer, args.patch, args.output)

if __name__ == "__main__":
    main()
