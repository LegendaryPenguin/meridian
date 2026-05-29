"""A deterministic pseudo-random generator, written out rather than imported.

`random.Random` is seeded reproducibly but its *stream* is an implementation detail:
CPython has changed the algorithm behind `randrange` before, and the documentation does
not promise the sequence is stable across versions. A corpus generator whose output is
only byte-identical on one interpreter build is not a reproducible corpus, and the whole
value of generating ground truth is that two people can regenerate the same ground truth.

SplitMix64 is a few lines of integer arithmetic with no floating point in the state, so
the stream is fixed by this file alone.
"""

from __future__ import annotations

import bisect
from typing import Sequence, TypeVar

_MASK = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15

T = TypeVar("T")


class Rng:
    """SplitMix64. Fast, well-distributed, and trivially portable."""

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        self._state = seed & _MASK

    def next_u64(self) -> int:
        self._state = (self._state + _GOLDEN) & _MASK
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return (z ^ (z >> 31)) & _MASK

    def random(self) -> float:
        # 53 significant bits, the most a float64 can hold without rounding changing the
        # value. Taking the low bits instead would be a measurably worse stream.
        return (self.next_u64() >> 11) / float(1 << 53)

    def below(self, bound: int) -> int:
        """Uniform integer in [0, bound).

        Rejection sampling rather than a modulo. A modulo biases the low values by
        roughly bound/2^64, which is unmeasurable here — but a biased draw in a corpus
        generator is the kind of thing that gets discovered halfway through explaining a
        result, and the loop costs one extra iteration in one case out of 2^64.
        """
        if bound <= 0:
            raise ValueError("bound must be positive")
        limit = _MASK - (_MASK % bound)
        while True:
            value = self.next_u64()
            if value <= limit:
                return value % bound

    def uniform(self, low: float, high: float) -> float:
        return low + (high - low) * self.random()

    def choice(self, items: Sequence[T]) -> T:
        return items[self.below(len(items))]

    def shuffled(self, items: Sequence[T]) -> list[T]:
        """Fisher-Yates over a copy, so the caller's sequence is untouched."""
        result = list(items)
        for i in range(len(result) - 1, 0, -1):
            j = self.below(i + 1)
            result[i], result[j] = result[j], result[i]
        return result

    def sample_cumulative(self, cumulative: Sequence[float]) -> int:
        """Index drawn from a distribution given as its cumulative weights.

        Inverse-CDF with a binary search: O(log n) per draw against O(n) for a linear
        scan, which matters because generating a corpus is millions of draws over a
        vocabulary of thousands.
        """
        target = self.random() * cumulative[-1]
        index = bisect.bisect_right(cumulative, target)
        return min(index, len(cumulative) - 1)


def derive(seed: int, *tags: object) -> int:
    """A sub-seed for one named part of the generation.

    Independent streams per concern — documents, queries, workloads — so that changing
    the number of documents does not shift the queries. Without this, every parameter
    change reshuffles everything downstream and two runs become incomparable for reasons
    that have nothing to do with what was changed.
    """
    value = seed & _MASK
    for tag in tags:
        text = str(tag).encode("utf-8")
        for byte in text:
            value = ((value ^ byte) * 0x100000001B3) & _MASK
        value = (value + _GOLDEN) & _MASK
    return value
