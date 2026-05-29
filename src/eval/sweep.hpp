#pragma once

// Parameter sweeps, and why every row is compared rather than ranked.
//
// A sweep's natural output is a table sorted by quality, and that is the thing to avoid:
// the top row of such a table is usually the largest candidate_k, because recall rises
// monotonically with it while latency rises faster. Sorting by NDCG recommends the
// slowest configuration in the sweep and shows no sign of having done so.
//
// So each row carries both axes and a `Comparison` against the baseline, and a row whose
// quality gain came with a p99 regression is labelled a tradeoff. The table's job is to
// make the decision visible, not to make it.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "harness.hpp"
#include "meridian/eval.hpp"
#include "meridian/retrieval.hpp"

namespace meridian::eval {

/// One configuration in a sweep.
struct SweepPoint {
    std::string label;

    /// The recall/latency dial. Swept because its effect on quality saturates well before
    /// its effect on latency does, and the saturation point is corpus-specific.
    uint32_t candidate_k = 200;

    /// ANN search effort. Not part of `Query`, so it is applied by the runner factory
    /// rather than carried into the query.
    uint32_t ef_search = 64;

    Bm25Params bm25;

    HybridMerger::Strategy strategy = HybridMerger::Strategy::kReciprocalRankFusion;
    double lexical_weight = 1.0;
    double semantic_weight = 1.0;

    /// Fills the parts of a query this point controls. The rest — ef_search, BM25
    /// parameters, the merge strategy — are configuration of the engine, not of the
    /// request, and reach it through the factory.
    void ApplyTo(Query* query) const;
};

const char* StrategyName(HybridMerger::Strategy strategy);

/// Derives a label from whatever the point has set, so an unlabelled sweep is still
/// readable.
std::string DescribePoint(const SweepPoint& point);

/// Builds a runner for one configuration. Returning a Result rather than a pointer
/// because building an index at a given ef_search can legitimately fail, and a sweep that
/// silently skipped a point would leave a hole in the table that looks like a gap in the
/// parameter range.
using RunnerFactory = std::function<Result<std::unique_ptr<QueryRunner>>(const SweepPoint&)>;

struct SweepRow {
    SweepPoint point;
    EvalRun run;

    /// Against the baseline row, always. Comparing each row to the previous one would
    /// make a sweep of ten points nine unrelated comparisons, and a sequence of
    /// individually-tolerable p99 increases is how a configuration ends up twice as slow
    /// with every step labelled "better".
    Comparison vs_baseline;
};

struct SweepResult {
    std::vector<SweepRow> rows;
    size_t baseline_index = 0;
    double latency_tolerance = 0.10;
};

struct SweepOptions {
    HarnessOptions harness;

    /// Which point is the baseline. Defaults to the first.
    size_t baseline_index = 0;

    /// Fractional p99 increase treated as material. Passed through to Compare.
    double latency_tolerance = 0.10;
};

Result<SweepResult> RunSweep(const std::vector<SweepPoint>& points, const RunnerFactory& factory,
                            const std::vector<JudgedQuery>& queries, const SweepOptions& options);

// ---------------------------------------------------------------------------
// Axis builders. Each varies one parameter and holds the rest at the base point, because
// a sweep that moves two parameters at once produces a table from which neither one's
// effect can be read.
// ---------------------------------------------------------------------------

std::vector<SweepPoint> SweepCandidateK(const SweepPoint& base, const std::vector<uint32_t>& values);
std::vector<SweepPoint> SweepEfSearch(const SweepPoint& base, const std::vector<uint32_t>& values);
std::vector<SweepPoint> SweepBm25(const SweepPoint& base, const std::vector<double>& k1_values,
                                  const std::vector<double>& b_values);
std::vector<SweepPoint> SweepHybrid(const SweepPoint& base,
                                    const std::vector<HybridMerger::Strategy>& strategies,
                                    const std::vector<std::pair<double, double>>& weights);

/// Renders the comparison table. Fixed-width columns rather than CSV: the point of this
/// output is that a human reads the quality and latency columns side by side and notices
/// the tradeoff rows.
std::string FormatTable(const SweepResult& result);

/// The per-row verdict explanations, which do not fit in a table cell and are the part
/// that carries both numbers.
std::string FormatVerdicts(const SweepResult& result);

}  // namespace meridian::eval
