#include "distance.hpp"

#include <cmath>

namespace meridian::ann {

// DOES THE COMPILER AUTOVECTORISE THE SCALAR VERSION AT -O2? Half of it, and the
// half it cannot do is the half that costs.
//
// Checked by reading the generated assembly rather than inferring it from the
// timing — though the two agree, which is the point of doing both:
//
//   c++ -std=c++20 -O2 -DNDEBUG -Iinclude -S -o - src/ann/distance.cpp
//
// On arm64 with Apple clang, the body of DotScalar's main loop is:
//
//   ldp q1, q2, [x9, #-32]      ; 16 floats loaded as four 128-bit vectors
//   fmul.4s v1, v1, v5          ; four products at a time, vectorised
//   mov s5, v1[3]               ; ...then every lane extracted back out
//   fadd s0, s0, s5             ; ...and summed one at a time into s0
//
// Sixteen `fmul.4s` lanes of real vector work, followed by sixteen *scalar*
// `fadd s0` instructions chained through one accumulator. The multiplies
// vectorise; the reduction does not, and it does not because it legally cannot:
// float addition is not associative, so summing in a different order gives a
// different answer, and -O2 does not include -ffast-math. The vectoriser is not
// being timid, it is being correct. The serial chain through s0 then runs at the
// latency of an FP add rather than its throughput, and that chain is the loop's
// critical path.
//
// The unrolled version sidesteps it by writing four accumulators in the source,
// which *is* a different (and explicit) summation order, so the compiler is free
// to keep four `fmadd`/`fmla` chains in flight. Measured in vector_test.cpp at
// dim=128, repeated across runs: about 2.0x on the dot product, 3.4x on squared L2,
// 2.4x on cosine. So the unrolling is worth keeping, and the reason it is worth
// keeping is not the unrolling — it is that only the author can authorise the
// reassociation, and the honest cost of authorising it is in
// ReassociationDivergesUnderCancellation.

float DotScalar(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

float L2SquaredScalar(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float NormSquaredScalar(const float* a, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += a[i] * a[i];
    return sum;
}

namespace {

// The four-accumulator shape is written out once per kernel rather than hidden
// behind a lambda or template, because the whole point is to be explicit about
// what the generated code should look like.
constexpr size_t kUnroll = 8;

}  // namespace

float DotUnrolled(const float* a, const float* b, size_t n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    const size_t blocks = n / kUnroll;
    size_t i = 0;
    for (size_t blk = 0; blk < blocks; ++blk, i += kUnroll) {
        s0 += a[i + 0] * b[i + 0];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        s0 += a[i + 4] * b[i + 4];
        s1 += a[i + 5] * b[i + 5];
        s2 += a[i + 6] * b[i + 6];
        s3 += a[i + 7] * b[i + 7];
    }
    // Tail. Folded into s0 so the partial-sum tree below is the same shape
    // regardless of whether n is a multiple of the unroll factor.
    for (; i < n; ++i) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

float L2SquaredUnrolled(const float* a, const float* b, size_t n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    const size_t blocks = n / kUnroll;
    size_t i = 0;
    for (size_t blk = 0; blk < blocks; ++blk, i += kUnroll) {
        const float d0 = a[i + 0] - b[i + 0];
        const float d1 = a[i + 1] - b[i + 1];
        const float d2 = a[i + 2] - b[i + 2];
        const float d3 = a[i + 3] - b[i + 3];
        const float d4 = a[i + 4] - b[i + 4];
        const float d5 = a[i + 5] - b[i + 5];
        const float d6 = a[i + 6] - b[i + 6];
        const float d7 = a[i + 7] - b[i + 7];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
        s0 += d4 * d4;
        s1 += d5 * d5;
        s2 += d6 * d6;
        s3 += d7 * d7;
    }
    for (; i < n; ++i) {
        const float d = a[i] - b[i];
        s0 += d * d;
    }
    return (s0 + s1) + (s2 + s3);
}

float NormSquaredUnrolled(const float* a, size_t n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    const size_t blocks = n / kUnroll;
    size_t i = 0;
    for (size_t blk = 0; blk < blocks; ++blk, i += kUnroll) {
        s0 += a[i + 0] * a[i + 0];
        s1 += a[i + 1] * a[i + 1];
        s2 += a[i + 2] * a[i + 2];
        s3 += a[i + 3] * a[i + 3];
        s0 += a[i + 4] * a[i + 4];
        s1 += a[i + 5] * a[i + 5];
        s2 += a[i + 6] * a[i + 6];
        s3 += a[i + 7] * a[i + 7];
    }
    for (; i < n; ++i) s0 += a[i] * a[i];
    return (s0 + s1) + (s2 + s3);
}

float Norm(const float* a, size_t n) { return std::sqrt(NormSquaredUnrolled(a, n)); }

float CosineWithNorms(const float* a, float norm_a, const float* b, float norm_b, size_t n) {
    const float denom = norm_a * norm_b;
    if (!(denom > 0.0f)) return 0.0f;  // see the header: no direction, no angle
    const float cos = DotUnrolled(a, b, n) / denom;
    // Rounding can push the ratio a hair outside [-1, 1] for near-parallel
    // vectors, and a cosine of 1.0000001 turns into a negative distance that
    // breaks the "distances are non-negative" assumption the graph relies on.
    if (cos > 1.0f) return 1.0f;
    if (cos < -1.0f) return -1.0f;
    return cos;
}

float CosineScalar(const float* a, const float* b, size_t n) {
    const float na = std::sqrt(NormSquaredScalar(a, n));
    const float nb = std::sqrt(NormSquaredScalar(b, n));
    const float denom = na * nb;
    if (!(denom > 0.0f)) return 0.0f;
    const float cos = DotScalar(a, b, n) / denom;
    if (cos > 1.0f) return 1.0f;
    if (cos < -1.0f) return -1.0f;
    return cos;
}

float CosineUnrolled(const float* a, const float* b, size_t n) {
    return CosineWithNorms(a, Norm(a, n), b, Norm(b, n), n);
}

float Distance(Metric metric, const float* a, float norm_a, const float* b, float norm_b,
               size_t n) {
    switch (metric) {
        case Metric::kCosine:
            return 1.0f - CosineWithNorms(a, norm_a, b, norm_b, n);
        case Metric::kInnerProduct:
            return -DotUnrolled(a, b, n);
        case Metric::kL2Squared:
            return L2SquaredUnrolled(a, b, n);
    }
    return 0.0f;  // unreachable for a valid enum; keeps -Wreturn-type quiet
}

float ScoreFromDistance(Metric metric, float distance) {
    switch (metric) {
        case Metric::kCosine:
            return 1.0f - distance;
        case Metric::kInnerProduct:
            return -distance;
        case Metric::kL2Squared:
            return -distance;
    }
    return 0.0f;
}

}  // namespace meridian::ann
