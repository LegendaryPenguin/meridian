// Distance kernel tests, and the benchmark that says what the unrolling is worth.
//
// The benchmark lives in the test binary rather than in bench/ on purpose: a kernel
// whose two implementations are supposed to agree numerically and whose timing is
// the reason both exist is one subject, and splitting it means the timing gets run
// once and then never again. ctest builds this at the project default
// (RelWithDebInfo, -O2), so the numbers mean something; run the binary directly, or
// ctest -V, to see them.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../src/ann/distance.hpp"
#include "testing.hpp"

using meridian::ann::CosineScalar;
using meridian::ann::CosineUnrolled;
using meridian::ann::CosineWithNorms;
using meridian::ann::Distance;
using meridian::ann::DotScalar;
using meridian::ann::DotUnrolled;
using meridian::ann::L2SquaredScalar;
using meridian::ann::L2SquaredUnrolled;
using meridian::ann::Metric;
using meridian::ann::Norm;
using meridian::ann::NormSquaredScalar;
using meridian::ann::NormSquaredUnrolled;
using meridian::ann::ScoreFromDistance;

namespace {

// A sanitiser build runs these loops roughly an order of magnitude slower, and the
// benchmark exists to compare two implementations against each other rather than to
// publish an absolute throughput, so the work is scaled down rather than skipped.
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define MERIDIAN_SLOW_BUILD 1
#endif
#endif
#ifndef MERIDIAN_SLOW_BUILD
#define MERIDIAN_SLOW_BUILD 0
#endif

class Rng {
public:
    explicit Rng(uint64_t seed) : state_(seed) {}
    uint64_t Next() {
        state_ += 0x9E3779B97F4A7C15ull;
        uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    /// Uniform in [-1, 1).
    float Signed() {
        return static_cast<float>(static_cast<double>(Next() >> 11) * (2.0 / 9007199254740992.0) -
                                  1.0);
    }
    std::vector<float> Vector(size_t n) {
        std::vector<float> v(n);
        for (float& x : v) x = Signed();
        return v;
    }

private:
    uint64_t state_;
};

/// Relative comparison, because the two implementations sum in different orders and
/// an absolute tolerance that works for a dot product of 0.01 is meaningless for one
/// of 1e30. Falls back to absolute near zero, where relative error is undefined.
bool Close(float a, float b, float rel = 2e-6f) {
    if (a == b) return true;
    const float scale = std::max(std::fabs(a), std::fabs(b));
    if (scale < 1e-30f) return std::fabs(a - b) < 1e-30f;
    return std::fabs(a - b) / scale <= rel;
}

double ElapsedNs(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
        .count();
}

}  // namespace

MERIDIAN_TEST(DotScalarAndUnrolledAgreeOnRandomVectors) {
    Rng rng(1);
    // Lengths deliberately straddle the unroll factor of eight, because the tail
    // loop is where an off-by-one in an unrolled kernel hides.
    for (size_t n : {1u, 2u, 3u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 64u, 127u, 128u, 129u, 1000u}) {
        const std::vector<float> a = rng.Vector(n);
        const std::vector<float> b = rng.Vector(n);
        const float s = DotScalar(a.data(), b.data(), n);
        const float u = DotUnrolled(a.data(), b.data(), n);
        if (!Close(s, u, 1e-5f)) {
            std::printf("  dot mismatch at n=%zu: scalar=%.9g unrolled=%.9g\n", n, s, u);
        }
        CHECK(Close(s, u, 1e-5f));

        const float ls = L2SquaredScalar(a.data(), b.data(), n);
        const float lu = L2SquaredUnrolled(a.data(), b.data(), n);
        CHECK(Close(ls, lu, 1e-5f));

        const float ns = NormSquaredScalar(a.data(), n);
        const float nu = NormSquaredUnrolled(a.data(), n);
        CHECK(Close(ns, nu, 1e-5f));

        CHECK(Close(CosineScalar(a.data(), b.data(), n), CosineUnrolled(a.data(), b.data(), n),
                    1e-5f));
    }
}

MERIDIAN_TEST(DotOfZeroLengthIsZero) {
    // n == 0 is reachable: a caller with an empty embedding should get a defined
    // answer rather than reading one float past nothing.
    CHECK_EQ(DotScalar(nullptr, nullptr, 0), 0.0f);
    CHECK_EQ(DotUnrolled(nullptr, nullptr, 0), 0.0f);
    CHECK_EQ(L2SquaredUnrolled(nullptr, nullptr, 0), 0.0f);
}

MERIDIAN_TEST(CosineOfAVectorWithItselfIsOne) {
    Rng rng(2);
    for (size_t n : {1u, 8u, 13u, 128u}) {
        const std::vector<float> a = rng.Vector(n);
        CHECK(Close(CosineScalar(a.data(), a.data(), n), 1.0f, 1e-5f));
        CHECK(Close(CosineUnrolled(a.data(), a.data(), n), 1.0f, 1e-5f));
    }
}

MERIDIAN_TEST(CosineOfIdenticalAndScaledVectorsIsOne) {
    // Cosine ignores magnitude by construction. Scaling one side by 1000 is the
    // cheapest way to notice a kernel that forgot to divide by a norm.
    const std::vector<float> a{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> b = a;
    for (float& x : b) x *= 1000.0f;
    CHECK(Close(CosineUnrolled(a.data(), b.data(), a.size()), 1.0f, 1e-5f));
}

MERIDIAN_TEST(CosineOfOrthogonalVectorsIsZero) {
    const std::vector<float> a{1.0f, 0.0f, 0.0f, 0.0f};
    const std::vector<float> b{0.0f, 1.0f, 0.0f, 0.0f};
    CHECK(Close(CosineScalar(a.data(), b.data(), 4), 0.0f));
    CHECK(Close(CosineUnrolled(a.data(), b.data(), 4), 0.0f));

    // And of an antiparallel pair, -1: cosine is signed, which is why cosine
    // *distance* runs to 2 rather than 1.
    std::vector<float> neg = a;
    for (float& x : neg) x = -x;
    CHECK(Close(CosineUnrolled(a.data(), neg.data(), 4), -1.0f, 1e-6f));
    CHECK(Close(Distance(Metric::kCosine, a.data(), Norm(a.data(), 4), neg.data(),
                         Norm(neg.data(), 4), 4),
                2.0f, 1e-6f));
}

MERIDIAN_TEST(CosineWithAZeroVectorIsZeroRatherThanNaN) {
    const std::vector<float> zero(16, 0.0f);
    Rng rng(3);
    const std::vector<float> a = rng.Vector(16);

    CHECK_EQ(CosineScalar(zero.data(), a.data(), 16), 0.0f);
    CHECK_EQ(CosineUnrolled(zero.data(), a.data(), 16), 0.0f);
    CHECK_EQ(CosineUnrolled(zero.data(), zero.data(), 16), 0.0f);
    CHECK_EQ(CosineWithNorms(zero.data(), 0.0f, a.data(), Norm(a.data(), 16), 16), 0.0f);

    // The property that actually matters downstream: the distance is a real number,
    // so it orders against other distances instead of comparing false to all of them.
    const float d = Distance(Metric::kCosine, zero.data(), 0.0f, a.data(), Norm(a.data(), 16), 16);
    CHECK(std::isfinite(d));
    CHECK_EQ(d, 1.0f);
}

MERIDIAN_TEST(L2OfIdenticalVectorsIsExactlyZero) {
    Rng rng(4);
    for (size_t n : {1u, 8u, 100u}) {
        const std::vector<float> a = rng.Vector(n);
        CHECK_EQ(L2SquaredScalar(a.data(), a.data(), n), 0.0f);
        CHECK_EQ(L2SquaredUnrolled(a.data(), a.data(), n), 0.0f);
    }
}

MERIDIAN_TEST(KernelsAgreeAtExtremeMagnitudes) {
    const size_t n = 100;

    // Large but not overflowing: 1e15 squared is 1e30, and a hundred of those is
    // 1e32, comfortably inside float's 3.4e38. Past about 1e19 per component
    // L2Squared overflows to infinity, which is a property of squaring in float and
    // is documented in distance.hpp rather than worked around.
    //
    // All one sign. An earlier version of this test alternated the sign of big_a,
    // which made the true dot product exactly zero by cancellation — and the two
    // implementations then disagreed, correctly, for the reason the next test is
    // about. A test of large magnitudes should test large magnitudes, not smuggle in
    // a cancellation case and blame the kernel.
    std::vector<float> big_a(n), big_b(n);
    for (size_t i = 0; i < n; ++i) {
        big_a[i] = 1e15f * static_cast<float>(1 + i % 3);
        big_b[i] = 2e15f;
    }
    CHECK(Close(DotScalar(big_a.data(), big_b.data(), n),
                DotUnrolled(big_a.data(), big_b.data(), n), 1e-5f));
    CHECK(Close(L2SquaredScalar(big_a.data(), big_b.data(), n),
                L2SquaredUnrolled(big_a.data(), big_b.data(), n), 1e-5f));
    // Cosine is scale-free, so it stays well-conditioned where L2 does not.
    CHECK(Close(CosineScalar(big_a.data(), big_b.data(), n),
                CosineUnrolled(big_a.data(), big_b.data(), n), 1e-5f));

    // Tiny: squares here land in the subnormal range and flush towards zero. Both
    // implementations must flush the same way, and cosine must still recover the
    // angle because the division cancels the scale.
    std::vector<float> small_a(n), small_b(n);
    for (size_t i = 0; i < n; ++i) {
        small_a[i] = 1e-20f;
        small_b[i] = 3e-20f;
    }
    CHECK(Close(DotScalar(small_a.data(), small_b.data(), n),
                DotUnrolled(small_a.data(), small_b.data(), n), 1e-5f));
    CHECK(Close(L2SquaredScalar(small_a.data(), small_b.data(), n),
                L2SquaredUnrolled(small_a.data(), small_b.data(), n), 1e-5f));
    CHECK(Close(CosineUnrolled(small_a.data(), small_b.data(), n), 1.0f, 1e-5f));
}

MERIDIAN_TEST(ReassociationDivergesUnderCancellation) {
    // The limit of the "both implementations agree" claim, stated rather than left to
    // be discovered.
    //
    // The unrolled kernel sums into four accumulators, which is a different
    // association of the same additions. Float addition is not associative, so the
    // two are not required to agree — and where the sum involves cancellation, they
    // genuinely do not. Here the terms are ±2e30 and the true dot product is exactly
    // zero: the scalar version alternates and cancels pair by pair, while the unrolled
    // version accumulates all the positives in two lanes and all the negatives in the
    // other two, then subtracts two enormous numbers at the end. Whatever survives is
    // rounding noise, and the two noises differ.
    //
    // This was found by the sanitiser build failing while the normal build passed: the
    // two had different FMA contraction, which was enough to change the residue. A
    // test that depends on which instructions the compiler picked is a test of the
    // compiler.
    //
    // So the tolerance here is relative to the size of the *terms*, not of the result,
    // which is the only meaningful accuracy claim available for a cancelling sum.
    const size_t n = 100;
    std::vector<float> a(n), b(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = 1e15f * (i % 2 == 0 ? 1.0f : -1.0f);
        b[i] = 2e15f;
    }
    const float term_magnitude = 2e30f;
    const float s = DotScalar(a.data(), b.data(), n);
    const float u = DotUnrolled(a.data(), b.data(), n);

    std::printf("  cancelling dot product (true value 0): scalar=%.6g unrolled=%.6g\n", s, u);

    // Both must be small compared to the terms being summed — that is the accuracy
    // guarantee a float reduction actually offers.
    const float bound = term_magnitude * static_cast<float>(n) * 1e-6f;
    CHECK(std::fabs(s) < bound);
    CHECK(std::fabs(u) < bound);

    // And cosine, which divides out the scale, must agree between the two to within
    // the same kind of bound rather than the usual relative one.
    CHECK(std::fabs(CosineScalar(a.data(), b.data(), n) - CosineUnrolled(a.data(), b.data(), n)) <
          1e-4f);
}

MERIDIAN_TEST(DistanceAndScoreRoundTripPerMetric) {
    Rng rng(5);
    const size_t n = 32;
    const std::vector<float> a = rng.Vector(n);
    const std::vector<float> b = rng.Vector(n);
    const float na = Norm(a.data(), n);
    const float nb = Norm(b.data(), n);

    const float cos = CosineUnrolled(a.data(), b.data(), n);
    CHECK(Close(ScoreFromDistance(Metric::kCosine,
                                  Distance(Metric::kCosine, a.data(), na, b.data(), nb, n)),
                cos, 1e-5f));

    const float dot = DotUnrolled(a.data(), b.data(), n);
    CHECK(Close(ScoreFromDistance(Metric::kInnerProduct,
                                  Distance(Metric::kInnerProduct, a.data(), na, b.data(), nb, n)),
                dot, 1e-5f));

    const float l2 = L2SquaredUnrolled(a.data(), b.data(), n);
    CHECK(Close(ScoreFromDistance(Metric::kL2Squared,
                                  Distance(Metric::kL2Squared, a.data(), na, b.data(), nb, n)),
                -l2, 1e-5f));
}

MERIDIAN_TEST(DistanceOrdersClosestFirst) {
    // The one property the graph code depends on: "smaller is closer", under every
    // metric. A sign error here would invert the whole search and still return
    // results.
    const std::vector<float> q{1.0f, 0.0f};
    const std::vector<float> near{0.9f, 0.1f};
    const std::vector<float> far{-1.0f, 0.2f};
    for (Metric m : {Metric::kCosine, Metric::kInnerProduct, Metric::kL2Squared}) {
        const float dn = Distance(m, q.data(), Norm(q.data(), 2), near.data(),
                                  Norm(near.data(), 2), 2);
        const float df =
            Distance(m, q.data(), Norm(q.data(), 2), far.data(), Norm(far.data(), 2), 2);
        CHECK(dn < df);
    }
}

MERIDIAN_TEST(BenchmarkScalarVersusUnrolled) {
    constexpr size_t kDim = 128;
    constexpr size_t kVectors = 2048;
    // 40 reps measured in about 15 ms, which is short enough that a scheduler hiccup
    // moved the ratio by 20% between runs. 200 reps costs a tenth of a second and the
    // ratio repeats to within a few percent.
    const int reps = MERIDIAN_SLOW_BUILD ? 10 : 200;

    Rng rng(0xBEEF);
    std::vector<float> data(kDim * kVectors);
    for (float& x : data) x = rng.Signed();
    const std::vector<float> query = rng.Vector(kDim);

    // Warm the data into cache first, so the first implementation measured is not
    // charged for the cold misses of the whole array.
    double warm = 0.0;
    for (size_t i = 0; i < kVectors; ++i) {
        warm += DotUnrolled(query.data(), data.data() + i * kDim, kDim);
    }
    if (warm == 1234567.0) std::printf("  (unreachable warm %f)\n", warm);

    struct Row {
        const char* name;
        float (*fn)(const float*, const float*, size_t);
    };
    const Row rows[] = {
        {"dot        scalar  ", &DotScalar},   {"dot        unrolled", &DotUnrolled},
        {"l2squared  scalar  ", &L2SquaredScalar}, {"l2squared  unrolled", &L2SquaredUnrolled},
        {"cosine     scalar  ", &CosineScalar}, {"cosine     unrolled", &CosineUnrolled},
    };

    std::printf("\n  distance kernels: dim=%zu, %zu vectors, %d reps (%s build)\n", kDim, kVectors,
                reps, MERIDIAN_SLOW_BUILD ? "sanitizer" : "optimized");
    std::printf("  %-20s %12s %12s\n", "kernel", "ns/call", "Mcalls/s");

    double ns_per_call[6] = {0, 0, 0, 0, 0, 0};
    for (size_t r = 0; r < 6; ++r) {
        double sink = 0.0;
        const auto start = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep) {
            for (size_t i = 0; i < kVectors; ++i) {
                sink += rows[r].fn(query.data(), data.data() + i * kDim, kDim);
            }
        }
        const double ns = ElapsedNs(start);
        // sink is consumed so the loop cannot be deleted as dead; without this the
        // benchmark reports the cost of an empty loop.
        if (sink == 1234567.0) std::printf("  (unreachable sink %f)\n", sink);
        ns_per_call[r] = ns / (double(reps) * double(kVectors));
        std::printf("  %-20s %12.2f %12.1f\n", rows[r].name, ns_per_call[r],
                    1000.0 / ns_per_call[r]);
    }

    std::printf("  unrolled speedup: dot %.2fx  l2squared %.2fx  cosine %.2fx\n",
                ns_per_call[0] / ns_per_call[1], ns_per_call[2] / ns_per_call[3],
                ns_per_call[4] / ns_per_call[5]);

    // Nothing is asserted about the ratio. The point of the measurement is to find
    // out what it is on this target, and a threshold here would turn a report into a
    // flaky test on a loaded machine. What is asserted is that the timing loop
    // actually ran.
    CHECK(ns_per_call[0] > 0.0);
    CHECK(ns_per_call[1] > 0.0);
}

MERIDIAN_TEST_MAIN()
