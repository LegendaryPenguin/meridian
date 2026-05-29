"""The generator's randomness has to be reproducible before anything built on it is."""

from __future__ import annotations

import os
import sys
import unittest

# The package lives in python/ and the tests in python/tests/, so unittest's discovery
# start directory is not on the import path. Added per file rather than through a shared
# helper, because importing the helper would have the same problem.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meridian_eval.rng import Rng, derive  # noqa: E402


class RngTest(unittest.TestCase):
    def test_same_seed_gives_the_same_stream(self) -> None:
        first = Rng(7)
        second = Rng(7)
        self.assertEqual(
            [first.next_u64() for _ in range(64)], [second.next_u64() for _ in range(64)]
        )

    def test_different_seeds_diverge_immediately(self) -> None:
        # Not a statistical claim, just that the seed reaches the first output. A generator
        # whose first draw ignores the low bits of the seed is a real mistake and it looks
        # fine until two configurations produce the same corpus.
        self.assertNotEqual(Rng(1).next_u64(), Rng(2).next_u64())
        self.assertNotEqual(Rng(0).next_u64(), Rng(1).next_u64())

    def test_stream_is_pinned_to_known_values(self) -> None:
        # SplitMix64 from seed 0 has published outputs. Pinning them here is what makes
        # "byte-identical corpus" a claim about this repository rather than about whichever
        # interpreter happens to run it.
        expected = [
            0xE220A8397B1DCDAF,
            0x6E789E6AA1B965F4,
            0x06C45D188009454F,
        ]
        rng = Rng(0)
        self.assertEqual([rng.next_u64() for _ in range(3)], expected)

    def test_below_stays_in_range(self) -> None:
        rng = Rng(99)
        values = [rng.below(7) for _ in range(2000)]
        self.assertTrue(all(0 <= v < 7 for v in values))
        # All seven values should appear in 2000 draws; a generator that cannot reach one of
        # them would quietly exclude a topic or a term.
        self.assertEqual(set(values), set(range(7)))

    def test_below_rejects_non_positive_bounds(self) -> None:
        with self.assertRaises(ValueError):
            Rng(1).below(0)

    def test_random_is_in_the_unit_interval(self) -> None:
        rng = Rng(5)
        values = [rng.random() for _ in range(5000)]
        self.assertTrue(all(0.0 <= v < 1.0 for v in values))
        # A mean far from 0.5 would mean the bit extraction is skewed.
        self.assertAlmostEqual(sum(values) / len(values), 0.5, delta=0.02)

    def test_shuffled_is_a_permutation_and_does_not_mutate(self) -> None:
        source = list(range(20))
        shuffled = Rng(3).shuffled(source)
        self.assertEqual(sorted(shuffled), source)
        self.assertEqual(source, list(range(20)))
        self.assertNotEqual(shuffled, source)

    def test_sample_cumulative_respects_the_weights(self) -> None:
        # Weights 100:1 over two outcomes. 3000 draws should land overwhelmingly on the
        # first; a reversed comparison inside the binary search would invert this.
        cumulative = (100.0, 101.0)
        rng = Rng(11)
        counts = [0, 0]
        for _ in range(3000):
            counts[rng.sample_cumulative(cumulative)] += 1
        self.assertGreater(counts[0], 2800)
        self.assertGreater(counts[1], 0)

    def test_derive_separates_streams_by_tag(self) -> None:
        # Independent streams per concern: changing the document count must not reshuffle
        # the queries.
        self.assertNotEqual(derive(1, "document", 0), derive(1, "query", 0))
        self.assertNotEqual(derive(1, "document", 0), derive(1, "document", 1))
        self.assertEqual(derive(1, "document", 0), derive(1, "document", 0))

    def test_derive_is_bounded_to_64_bits(self) -> None:
        self.assertLess(derive(2**63, "x", "y"), 2**64)


if __name__ == "__main__":
    unittest.main()
