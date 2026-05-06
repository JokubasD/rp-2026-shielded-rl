import unittest

class TestFailing(unittest.TestCase):
    def test_fail(self):
        self.assertEqual(0, 1)