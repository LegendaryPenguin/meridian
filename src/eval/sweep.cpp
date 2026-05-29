#include "sweep.hpp"

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace meridian::eval {
namespace {

std::string Fixed(double value, int precision) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*f", precision, value);
    return buffer;
}

/// Microseconds with two decimals. Nanoseconds are what the engine records and what a
/// comparison should use; microseconds are what a reader can hold in their head.
std::string Micros(uint64_t nanos) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.2f", static_cast<double>(nanos) / 1000.0);
    return buffer;
}

std::string Pad(const std::string& text, size_t width, bool right_align) {
    if (text.size() >= width) return text;
    const std::string padding(width - text.size(), ' ');
    return right_align ? padding + text : text + padding;
}

}  // namespace

void SweepPoint::ApplyTo(Query* query) const { query->candidate_k = candidate_k; }

const char* StrategyName(HybridMerger::Strategy strategy) {
    switch (strategy) {
        case HybridMerger::Strategy::kReciprocalRankFusion: return "rrf";
        case HybridMerger::Strategy::kNormalizedLinear: return "norm-linear";
        case HybridMerger::Strategy::kZScoreLinear: return "zscore-linear";
    }
    return "unknown";
}

std::string DescribePoint(const SweepPoint& point) {
    if (!point.label.empty()) return point.label;
    std::ostringstream out;
    out << "ck=" << point.candidate_k << " ef=" << point.ef_search
        << " k1=" << Fixed(point.bm25.k1, 2) << " b=" << Fixed(point.bm25.b, 2) << " "
        << StrategyName(point.strategy) << " w=" << Fixed(point.lexical_weight, 2) << "/"
        << Fixed(point.semantic_weight, 2);
    return out.str();
}

Result<SweepResult> RunSweep(const std::vector<SweepPoint>& points, const RunnerFactory& factory,
                            const std::vector<JudgedQuery>& queries, const SweepOptions& options) {
    if (points.empty()) return Status::InvalidArgument("sweep needs at least one point");
    if (options.baseline_index >= points.size()) {
        return Status::InvalidArgument("sweep baseline index " +
                                       std::to_string(options.baseline_index) + " is out of range");
    }
    if (!factory) return Status::InvalidArgument("sweep needs a runner factory");

    SweepResult result;
    result.baseline_index = options.baseline_index;
    result.latency_tolerance = options.latency_tolerance;
    result.rows.reserve(points.size());

    for (const auto& point : points) {
        MERIDIAN_ASSIGN(std::unique_ptr<QueryRunner> runner, factory(point));
        if (!runner) {
            return Status::Internal("sweep factory returned a null runner for " +
                                    DescribePoint(point));
        }

        HarnessOptions harness = options.harness;
        harness.candidate_k = point.candidate_k;
        if (harness.label.empty()) harness.label = DescribePoint(point);

        MERIDIAN_ASSIGN(EvalRun run, RunHarness(*runner, queries, harness));
        SweepRow row;
        row.point = point;
        row.run = std::move(run);
        result.rows.push_back(std::move(row));
    }

    const EvalRun& baseline = result.rows[options.baseline_index].run;
    for (auto& row : result.rows) {
        row.vs_baseline = Compare(baseline, row.run, options.latency_tolerance);
    }
    return result;
}

std::vector<SweepPoint> SweepCandidateK(const SweepPoint& base,
                                        const std::vector<uint32_t>& values) {
    std::vector<SweepPoint> points;
    points.reserve(values.size());
    for (uint32_t value : values) {
        SweepPoint point = base;
        point.candidate_k = value;
        point.label = "candidate_k=" + std::to_string(value);
        points.push_back(point);
    }
    return points;
}

std::vector<SweepPoint> SweepEfSearch(const SweepPoint& base, const std::vector<uint32_t>& values) {
    std::vector<SweepPoint> points;
    points.reserve(values.size());
    for (uint32_t value : values) {
        SweepPoint point = base;
        point.ef_search = value;
        point.label = "ef_search=" + std::to_string(value);
        points.push_back(point);
    }
    return points;
}

std::vector<SweepPoint> SweepBm25(const SweepPoint& base, const std::vector<double>& k1_values,
                                  const std::vector<double>& b_values) {
    std::vector<SweepPoint> points;
    points.reserve(k1_values.size() * b_values.size());
    for (double k1 : k1_values) {
        for (double b : b_values) {
            SweepPoint point = base;
            point.bm25.k1 = k1;
            point.bm25.b = b;
            point.label = "k1=" + Fixed(k1, 2) + ",b=" + Fixed(b, 2);
            points.push_back(point);
        }
    }
    return points;
}

std::vector<SweepPoint> SweepHybrid(const SweepPoint& base,
                                    const std::vector<HybridMerger::Strategy>& strategies,
                                    const std::vector<std::pair<double, double>>& weights) {
    std::vector<SweepPoint> points;
    points.reserve(strategies.size() * weights.size());
    for (auto strategy : strategies) {
        for (const auto& weight : weights) {
            SweepPoint point = base;
            point.strategy = strategy;
            point.lexical_weight = weight.first;
            point.semantic_weight = weight.second;
            point.label = std::string(StrategyName(strategy)) + " " + Fixed(weight.first, 2) + "/" +
                          Fixed(weight.second, 2);
            points.push_back(point);
        }
    }
    return points;
}

std::string FormatTable(const SweepResult& result) {
    struct Column {
        std::string header;
        bool right_align;
        std::vector<std::string> cells;
    };

    std::vector<Column> columns = {
        {"configuration", false, {}}, {"NDCG@k", true, {}},   {"P@k", true, {}},
        {"recall@k", true, {}},       {"MRR", true, {}},      {"unjudged", true, {}},
        {"p50 us", true, {}},         {"p95 us", true, {}},   {"p99 us", true, {}},
        {"cands", true, {}},          {"timeouts", true, {}}, {"verdict", false, {}},
    };

    for (size_t i = 0; i < result.rows.size(); ++i) {
        const SweepRow& row = result.rows[i];
        const bool is_baseline = i == result.baseline_index;
        std::string name = DescribePoint(row.point);
        if (is_baseline) name += " (baseline)";

        size_t c = 0;
        columns[c++].cells.push_back(name);
        columns[c++].cells.push_back(Fixed(row.run.quality.ndcg_at_k, 4));
        columns[c++].cells.push_back(Fixed(row.run.quality.precision_at_k, 4));
        columns[c++].cells.push_back(Fixed(row.run.quality.recall_at_k, 4));
        columns[c++].cells.push_back(Fixed(row.run.quality.mrr, 4));
        columns[c++].cells.push_back(std::to_string(row.run.quality.unjudged_in_top_k));
        columns[c++].cells.push_back(Micros(row.run.latency.p50_ns));
        columns[c++].cells.push_back(Micros(row.run.latency.p95_ns));
        columns[c++].cells.push_back(Micros(row.run.latency.p99_ns));
        columns[c++].cells.push_back(Fixed(row.run.mean_candidates, 1));
        columns[c++].cells.push_back(std::to_string(row.run.queries_timed_out));
        columns[c++].cells.push_back(is_baseline ? "-" : VerdictName(row.vs_baseline.verdict));
    }

    std::vector<size_t> widths;
    widths.reserve(columns.size());
    for (const auto& column : columns) {
        size_t width = column.header.size();
        for (const auto& cell : column.cells) width = std::max(width, cell.size());
        widths.push_back(width);
    }

    std::ostringstream out;
    for (size_t c = 0; c < columns.size(); ++c) {
        out << (c == 0 ? "" : "  ") << Pad(columns[c].header, widths[c], columns[c].right_align);
    }
    out << "\n";
    for (size_t c = 0; c < columns.size(); ++c) {
        out << (c == 0 ? "" : "  ") << std::string(widths[c], '-');
    }
    out << "\n";
    for (size_t r = 0; r < result.rows.size(); ++r) {
        for (size_t c = 0; c < columns.size(); ++c) {
            out << (c == 0 ? "" : "  ") << Pad(columns[c].cells[r], widths[c], columns[c].right_align);
        }
        out << "\n";
    }
    return out.str();
}

std::string FormatVerdicts(const SweepResult& result) {
    std::ostringstream out;
    for (size_t i = 0; i < result.rows.size(); ++i) {
        if (i == result.baseline_index) continue;
        out << "  " << DescribePoint(result.rows[i].point) << ": "
            << result.rows[i].vs_baseline.explanation << "\n";
    }
    return out.str();
}

}  // namespace meridian::eval
