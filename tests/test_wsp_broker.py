import json
import tempfile
import unittest
from pathlib import Path
from wsp_broker import Budget, BudgetExceeded, InvalidQuery, QueryBroker, verify_ledger

class MockBackend:
    def count_tokens(self, text):
        return len(text.split()) or 1
    def count_chat_tokens(self, chat):
        return sum(self.count_tokens(m["content"]) + 4 for m in chat)
    def execute(self, req):
        return {"status": "ok", "generated_tokens": 15, "backend_seconds": 0.02}

class BrokerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "log.jsonl"
        self.broker = QueryBroker(MockBackend(), self.path, Budget(passes=2, python_calls=1))
    def test_one_and_budget(self):
        self.broker.query({"type": "generate", "model": "both", "problems": ["gsm8k-1"], "n": 1})
        self.broker.query({"type": "teacher_forced", "model": "original", "texts": ["hello world"]})
        self.assertEqual(verify_ledger(self.path), (2,30))
        with self.assertRaises(BudgetExceeded):
            self.broker.query({"type": "generate", "model": "original", "prompts": ["test"]})
    def test_pair_cap_is_per_model(self):
        with self.assertRaises(InvalidQuery):
            self.broker.query({"type": "generate", "model": "both", "prompts": ["p"]*257})
        self.assertEqual(self.broker.used, 0)
    def test_single_cap(self):
        with self.assertRaises(InvalidQuery):
            self.broker.query({"type": "generate", "model": "edited", "prompts": ["p"]*513})
    def test_no_free_empty_queries(self):
        with self.assertRaises(InvalidQuery):
            self.broker.query({"type": "teacher_forced", "model": "original", "texts": []})
        with self.assertRaises(InvalidQuery):
            self.broker.query({"type": "generate", "model": "original", "prompts": []})
        self.assertEqual(self.broker.used, 0)
    def test_teacher_token_cap(self):
        with self.assertRaises(InvalidQuery):
            self.broker.query({"type": "teacher_forced", "model": "original", "texts": ["a " * 32769]})
        self.assertEqual(self.broker.used, 0)
    def test_failed_backend_not_counted(self):
        class FailureBackend(MockBackend):
            def execute(self, req):
                raise RuntimeError("unavailable")
        broker = QueryBroker(FailureBackend(), Path(self.temp.name) / "bad.jsonl")
        with self.assertRaises(RuntimeError):
            broker.query({"type": "generate", "model": "original", "prompts": ["hello"]})
        self.assertEqual(broker.used, 0)
    def test_ledger_tamper_detect(self):
        self.broker.query({"type": "generate", "model": "original", "prompts": ["hello"]})
        obj = json.loads(self.path.read_text().strip())
        obj["generated_tokens"] = 9000
        self.path.write_text(json.dumps(obj) + "\n")
        with self.assertRaises(ValueError):
            verify_ledger(self.path)
    def test_python_budget(self):
        self.broker.record_python("a"*64)
        with self.assertRaises(BudgetExceeded):
            self.broker.record_python("b"*64)

if __name__ == "__main__":
    unittest.main()
