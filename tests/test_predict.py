import unittest
from src.inference.predict import predict

class TestPredict(unittest.TestCase):
    def test_neutral(self):
        result = predict("Hello world")
        self.assertEqual(result['neutral']['pred'], 1)

    def test_toxic(self):
        result = predict("You idiot")
        self.assertEqual(result['toxic']['pred'], 1)

if __name__ == '__main__':
    unittest.main()