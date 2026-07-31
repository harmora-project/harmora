from __future__ import annotations

import unittest

from harmora_downstream.sampling import stable_hash


class SamplingTests(unittest.TestCase):
    def test_stable_hash_dict_order(self):
        self.assertEqual(stable_hash({"a": 1, "b": 2}), stable_hash({"b": 2, "a": 1}))

    def test_hash_changes_with_content(self):
        self.assertNotEqual(stable_hash([1, 2, 3]), stable_hash([1, 2, 4]))


if __name__ == "__main__":
    unittest.main()
