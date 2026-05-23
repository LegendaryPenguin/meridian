#pragma once

// Vector distance kernels.
//
// This is the innermost loop of the whole semantic path: an HNSW search at
// ef_search=200 over 128-dimensional vectors spends essentially all of its time
// here, so the file exists twice over — once written the obvious way and once
// unrolled — and vector_test.cpp times both. The unrolling is not assumed to be
// worth anything; the measurement decides, and the numbers are in the test output.
//
// Everything here takes raw pointers plus a length rather than std::span. The
// callers hold one big contiguous float array and index into it, and a span would
// be constructed and destructured on every one of millions of calls for no
// expressive gain at this level.

#include <cstddef>
#include <cstdint>

namespace meridian::ann {

/// Which notion of "close" an index uses.
///
/// Kept separate from the distance functions because the choice is a property of
/// the embedding model, not of the index: vectors from a model trained with a
/// cosine objective are not meaningfully comparable under L2 and vice versa.
enum class Metric : uint8_t {
    kCosine,        // angle only; magnitude ignored
    kInnerProduct,  // magnitude counts, so longer vectors win
    kL2Squared,     // the square root is never taken — see below
};

// --- scalar reference implementations ------------------------------------------
//
// These define the expected answer. The unrolled versions are checked against
// them, not the other way round.

float DotScalar(const float* a, const float* b, size_t n);
float L2SquaredScalar(const float* a, const float* b, size_t n);
float NormSquaredScalar(const float* a, size_t n);

// --- explicitly unrolled implementations ---------------------------------------
//
// Unrolled by eight with four independent accumulators. The accumulator count is
// the point, more than the unrolling: a single running sum makes each add wait on
// the previous one, so the loop runs at the latency of an FP add rather than its
// throughput. Four partial sums let four adds be in flight at once.
//
// Summing in a different order means these do not return bit-identical results to
// the scalar versions. For ordinary embedding data the difference is a few ulp and
// the tests compare with a relative tolerance. Where the sum cancels — terms of
// both signs summing to near zero — the difference is unbounded in relative terms,
// because what survives is rounding noise and the two orders produce different
// noise. Nothing in the engine depends on that case: a cancelling dot product means
// two orthogonal vectors, and their *ordering* against other vectors is unaffected
// by which near-zero value is reported. vector_test.cpp measures it anyway, so the
// limit of the agreement is written down rather than discovered later.

float DotUnrolled(const float* a, const float* b, size_t n);
float L2SquaredUnrolled(const float* a, const float* b, size_t n);
float NormSquaredUnrolled(const float* a, size_t n);

/// Euclidean norm. Used to precompute per-vector norms so cosine costs one dot
/// product at query time instead of three.
float Norm(const float* a, size_t n);

/// Cosine similarity, computing both norms on the spot. Convenient, and roughly
/// three times the work of the precomputed-norm form.
float CosineScalar(const float* a, const float* b, size_t n);
float CosineUnrolled(const float* a, const float* b, size_t n);

/// Cosine similarity given norms the caller already has.
///
/// A zero vector has no direction, so there is no angle to report. Returning 0
/// rather than NaN is a deliberate choice: NaN compares false against everything,
/// which would silently corrupt every heap ordering it touched and produce a
/// result list that is wrong without being empty. 0 is the value that says "no
/// evidence of similarity", which is the honest answer for a vector with no
/// direction.
float CosineWithNorms(const float* a, float norm_a, const float* b, float norm_b, size_t n);

/// Distance under `metric`: smaller is closer, always.
///
/// Normalising the direction of comparison here is what lets the graph code have
/// exactly one ordering and no per-metric branches in the heaps.
///
///   kCosine        1 - cos        in [0, 2]
///   kInnerProduct  -dot           unbounded, and not a metric at all: it
///                                 violates the triangle inequality, so an
///                                 inner-product graph is navigable only
///                                 empirically. Stated because it is a real
///                                 caveat on recall, not a footnote.
///   kL2Squared     |a-b|^2        monotone in |a-b|, so the square root would
///                                 change no ordering and cost a sqrt per
///                                 comparison. It does mean the value overflows
///                                 float for component magnitudes past ~1e19.
float Distance(Metric metric, const float* a, float norm_a, const float* b, float norm_b,
               size_t n);

/// The similarity to report to a caller, given an internal distance. Candidate
/// scores are "higher is better" throughout the rest of the engine, so the sign
/// flip happens once, here.
float ScoreFromDistance(Metric metric, float distance);

}  // namespace meridian::ann
