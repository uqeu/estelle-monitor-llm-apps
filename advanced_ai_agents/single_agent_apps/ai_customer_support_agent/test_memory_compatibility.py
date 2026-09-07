"""Real local Qdrant checks for the Mem0 version used by this application."""
import os
import tempfile
import unittest

os.environ.setdefault("MEM0_TELEMETRY", "false")
os.environ.setdefault("MEM0_DIR", tempfile.mkdtemp(prefix="support-mem0-test-"))

from mem0.vector_stores.qdrant import Qdrant


class MemoryCompatibilityTests(unittest.TestCase):
    def test_search_retrieves_only_the_requested_customer(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Qdrant("support", 3, path=directory, on_disk=True)
            try:
                store.insert([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    payloads=[{"user_id": "alice", "memory": "order A"},
                              {"user_id": "bob", "memory": "order B"}], ids=[1, 2])
                hits = store.search([1.0, 0.0, 0.0], filters={"user_id": "alice"})
                self.assertEqual([(hit.id, hit.payload["memory"]) for hit in hits], [(1, "order A")])
                self.assertEqual(store.search([1.0, 0.0, 0.0], filters={"user_id": "absent"}), [])
            finally:
                store.client.close()


if __name__ == "__main__":
    unittest.main()
