"""Qwen3.5-0.8B pilot evaluator. Requires CUDA and optional dependencies.

Not an exact reproduction of the MIT vLLM / presence-penalty protocol.
Use *identical* arguments for original and edited checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def last_boxed(s: str) -> str | None:
    """Extract the last balanced LaTeX \boxed{...}, including nested braces."""
    needle = "\\boxed{"
    start = s.rfind(needle)
    if start < 0:
        return None
    depth = 1
    begin = start + len(needle)
    for i in range(begin, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[begin:i]
    return None


def is_correct(gold: str, prediction: str) -> bool:
    from math_verify import parse, verify
    target = last_boxed(prediction)
    if target is None:
        return False
    g = parse("$" + gold.strip().strip("$") + "$")
    p = parse("$" + target + "$")
    return bool(g and p and verify(g, p))


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen mathematical evaluation (pilot)")
    p.add_argument("--model", required=True, help="Hugging Face model id or local checkpoint path")
    p.add_argument("--dataset", type=Path, required=True, help="JSONL with id, question, answer")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--limit", type=int, default=0, help="0 for all rows")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()
    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        import math_verify  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Install optional GPU dependencies: torch transformers accelerate math-verify[antlr4_13_2]"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("This pilot requires a CUDA GPU; CPU execution is intentionally disabled.")
    if args.temperature <= 0 or args.max_new_tokens <= 0:
        raise SystemExit("Temperature and max-new-tokens must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=False
    ).eval()
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("Empty dataset")
    outputs = []
    for idx, row in enumerate(rows):
        prompt = row["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", enable_thinking=args.thinking,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            ids = model.generate(
                **inputs, do_sample=True, temperature=args.temperature,
                top_p=args.top_p, top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        fresh = ids[0][inputs["input_ids"].shape[-1]:]
        response = processor.tokenizer.decode(fresh, skip_special_tokens=True)
        truncated = (
            len(fresh) >= args.max_new_tokens
            and int(fresh[-1]) != processor.tokenizer.eos_token_id
        )
        outputs.append({
            "id": row["id"], "correct": is_correct(str(row["answer"]), response),
            "output_tokens": len(fresh), "truncated": truncated,
            "boxed_found": last_boxed(response) is not None,
            "response": response,
        })
        print(f"{idx + 1}/{len(rows)} {row['id']} correct={outputs[-1]['correct']}", flush=True)

    result = {
        "dataset_sha256": digest(args.dataset), "model": args.model, "seed": args.seed,
        "thinking": args.thinking, "temperature": args.temperature,
        "top_p": args.top_p, "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens, "n": len(outputs),
        "accuracy": sum(r["correct"] for r in outputs) / len(outputs),
        "boxed_completion_rate": sum(r["boxed_found"] for r in outputs) / len(outputs),
        "truncation_rate": sum(r["truncated"] for r in outputs) / len(outputs),
        "mean_output_tokens": sum(r["output_tokens"] for r in outputs) / len(outputs),
        "protocol": "local_pilot_not_MIT_reproduction", "outputs": outputs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"accuracy={result['accuracy']:.3f} truncation={result['truncation_rate']:.3f}")


if __name__ == "__main__":
    main()
