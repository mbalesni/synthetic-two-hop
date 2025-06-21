import unittest

from latent_reasoning.datagen.synthetic_spouses.generate import create_derangement


class TestDerangement(unittest.TestCase):
    def test_create_derangement(self):
        # Number of times to repeat each test
        num_repetitions = 100

        for _ in range(num_repetitions):
            # Test with a list of integers
            original = list(range(100))
            deranged = create_derangement(original)

            # Check that the derangement has the same length as the original
            self.assertEqual(len(original), len(deranged))

            # Check that all elements are present in the derangement
            self.assertEqual(set(original), set(deranged))

            # Check that no element is in its original position
            self.assertTrue(all(o != d for o, d in zip(original, deranged)))

        for _ in range(num_repetitions):
            # Test with a small list to ensure edge cases are handled
            small_original = [1, 2, 3]
            small_deranged = create_derangement(small_original)
            self.assertEqual(len(small_original), len(small_deranged))
            self.assertEqual(set(small_original), set(small_deranged))
            self.assertTrue(all(o != d for o, d in zip(small_original, small_deranged)))

    def test_create_derangement_randomness(self):
        # Test that multiple calls produce different results
        original = list(range(10))
        derangements = [create_derangement(original) for _ in range(100)]

        # Check that we have at least 2 unique derangements
        # (it's extremely unlikely to get the same derangement 100 times)
        self.assertTrue(len(set(tuple(d) for d in derangements)) > 1)


if __name__ == "__main__":
    unittest.main()
