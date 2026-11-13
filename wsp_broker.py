"""MIT-inspired query-budget broker; NOT a secure isolation boundary.

A controlled, independently hosted backend is required for real evaluations.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

@dataclass(frozen=True)
class Budget:
    passes: int = 50
    python_calls: int = 300
    max_generate: int = 512
    max_paired_per_model: int = 256
    max_teacher_texts: int = 1024
    max_teacher_tokens: int = 524288
    max_text_tokens: int = 32768

class Backend(Protocol):
    def count_tokens(self, text: str) -> int: ...
    def execute(self, request: dict[str, Any]) -> dict[str, Any]: ...

class InvalidQuery(ValueError):
    pass

class BudgetExceeded(RuntimeError):
    pass

class QueryBroker:
    """Single-writer in-memory budget and append-only, hash-chained JSONL audit.
    Must be isolated from Agent and externally synchronized for formal use.
    """
    def __init__(self, backend: Backend, ledger: Path, budget: Budget = Budget()):
        if budget.passes <= 0:
            raise ValueError("passes must be positive")
        self.backend = backend
        self.ledger = ledger
        self.budget = budget
        self.used = 0
        self.python_used = 0
        self.prev_hash = "0" * 64
        if ledger.exists() and ledger.stat().st_size:
            raise ValueError("ledger must be absent or empty for a new run")
        ledger.parent.mkdir(parents=True, exist_ok=True)

    def _validate(self, req: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(req, dict):
            raise InvalidQuery("request must be a JSON object")
        kind = req.get("type")
        model = req.get("model")
        if kind not in ("generate", "teacher_forced"):
            raise InvalidQuery("invalid request type")
        if kind == "generate":
            if model not in ("original", "edited", "both"):
                raise InvalidQuery("invalid generation model")
            n = req.get("n", 1)
            if type(n) is not int or n <= 0:
                raise InvalidQuery("n must be a positive integer")
            problems = req.get("problems", [])
            prompts = req.get("prompts", [])
            if not isinstance(problems, list) or not isinstance(prompts, list):
                raise InvalidQuery("problems and prompts must be lists")
            if any(not isinstance(x, str) or not x for x in problems + prompts):
                raise InvalidQuery("all prompts / ids must be nonempty strings")
            total = (len(problems) + len(prompts)) * n
            if total == 0:
                raise InvalidQuery("generation cannot be empty")
            limit = self.budget.max_paired_per_model if model == "both" else self.budget.max_generate
            if total > limit:
                raise InvalidQuery("generation exceeds per-pass sequence cap")
            max_tokens = req.get("max_tokens", 32768)
            if type(max_tokens) is not int or not 1 <= max_tokens <= self.budget.max_text_tokens:
                raise InvalidQuery("max_tokens out of range")
            return {"type": kind, "model": model, "sequences_per_model": total,
                    "models_evaluated": 2 if model == "both" else 1, "max_tokens": max_tokens}
        if model not in ("original", "edited"):
            raise InvalidQuery("teacher_forced: original or edited only")
        texts, chats = req.get("texts"), req.get("chats")
        if (texts is None) == (chats is None):
            raise InvalidQuery("provide exactly one of texts / chats")
        if texts is not None:
            if not isinstance(texts, list) or any(not isinstance(x, str) for x in texts):
                raise InvalidQuery("texts must be strings")
            normalized = texts
        else:
            if not isinstance(chats, list) or any(
                not isinstance(chat, list) or not all(
                    isinstance(msg, dict) and isinstance(msg.get("content"), str) for msg in chat
                ) for chat in chats
            ):
                raise InvalidQuery("chats must be lists of text messages")
            normalized = ["".join(m["content"] for m in chat) for chat in chats]
        if not normalized or len(normalized) > self.budget.max_teacher_texts:
            raise InvalidQuery("too many or zero teacher texts")
        if chats is not None:
            counter = getattr(self.backend, "count_chat_tokens", None)
            if counter is None:
                raise InvalidQuery("backend must count chat-template tokens")
            counts = [counter(chat) for chat in chats]
        else:
            counts = [self.backend.count_tokens(s) for s in normalized]
        if any(type(c) is not int or not 1 <= c <= self.budget.max_text_tokens for c in counts):
            raise InvalidQuery("invalid token count per text")
        if sum(counts) > self.budget.max_teacher_tokens:
            raise InvalidQuery("teacher-forced input token cap exceeded")
        return {"type": kind, "model": model, "texts": len(normalized), "input_tokens": sum(counts)}

    def query(self, req: dict[str, Any]) -> dict[str, Any]:
        if self.used >= self.budget.passes:
            raise BudgetExceeded("forward pass budget exhausted")
        metadata = self._validate(req)
        result = self.backend.execute(req)
        if not isinstance(result, dict) or result.get("status") != "ok":
            raise RuntimeError("backend did not return status=ok; pass not billed")
        tokens = result.get("generated_tokens")
        if type(tokens) is not int or tokens < 0:
            raise RuntimeError("backend must report measured generated_tokens")
        self.used += 1
        event = {"schema": 1, "call_id": self.used, "query": metadata,
                 "generated_tokens": tokens,
                 "backend_seconds": result.get("backend_seconds"),
                 "prev_hash": self.prev_hash}
        payload = json.dumps(event, sort_keys=True, separators=(",", ":"))
        event["hash"] = hashlib.sha256(payload.encode()).hexdigest()
        with self.ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
            f.flush()
        self.prev_hash = event["hash"]
        return {"call_id": self.used, "remaining_passes": self.budget.passes - self.used,
                "result": result}

    def record_python(self, code_digest: str) -> None:
        if self.python_used >= self.budget.python_calls:
            raise BudgetExceeded("Python budget exhausted")
        if not isinstance(code_digest, str) or len(code_digest) != 64:
            raise InvalidQuery("code digest must be sha256 hex string")
        self.python_used += 1

def verify_ledger(path: Path) -> tuple[int, int]:
    previous = "0" * 64
    count = 0
    tokens = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        found = record.pop("hash", None)
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(payload.encode()).hexdigest()
        if record["prev_hash"] != previous or found != expected or record["call_id"] != count + 1:
            raise ValueError(f"invalid ledger hash chain at call {count+1}")
        previous = found
        count += 1
        tokens += record["generated_tokens"]
    return count, tokens
