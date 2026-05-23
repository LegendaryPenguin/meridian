#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace meridian {

// Status carries a failure reason instead of an exception.
//
// Exceptions are avoided on the serving path deliberately. A search engine fails for
// ordinary, expected reasons — an unparseable query, a segment mid-merge, a deadline
// that expired — and those are control flow, not emergencies. Making them values puts
// them in the signatures, which matters most on the retrieval path where a missed
// error becomes a silently short result list rather than a crash. A query that
// returns 8 results instead of 10 because a segment failed to open is the worst
// failure mode here: nothing looks wrong.
enum class Code {
    kOk = 0,
    kQueryError,       // the query text cannot be parsed
    kNotFound,         // no such term, document, segment or index generation
    kNotImplemented,   // deliberately outside scope
    kIoError,          // the file system said no
    kCorruption,       // on-disk bytes failed a check
    kInvalidArgument,
    kDeadlineExceeded, // the query ran out of its latency budget
    kOverloaded,       // refused rather than queued; see serve/
    kInternal,         // an invariant this code owns was violated
};

class Status {
public:
    Status() = default;

    static Status Ok() { return {}; }
    static Status QueryError(std::string m) { return {Code::kQueryError, std::move(m)}; }
    static Status NotFound(std::string m) { return {Code::kNotFound, std::move(m)}; }
    static Status DeadlineExceeded(std::string m) { return {Code::kDeadlineExceeded, std::move(m)}; }
    static Status Overloaded(std::string m) { return {Code::kOverloaded, std::move(m)}; }
    static Status NotImplemented(std::string m) { return {Code::kNotImplemented, std::move(m)}; }
    static Status IoError(std::string m) { return {Code::kIoError, std::move(m)}; }
    static Status Corruption(std::string m) { return {Code::kCorruption, std::move(m)}; }
    static Status InvalidArgument(std::string m) { return {Code::kInvalidArgument, std::move(m)}; }
    static Status Internal(std::string m) { return {Code::kInternal, std::move(m)}; }

    bool ok() const { return code_ == Code::kOk; }
    Code code() const { return code_; }
    const std::string& message() const { return message_; }

    std::string ToString() const;

    /// The same failure with `where` prepended to the message.
    ///
    /// Keeps the code, which is the whole reason this exists rather than rebuilding the
    /// status at the call site. A layer adding context has to re-pick a factory to do that,
    /// and re-picking means guessing: a corrupt posting block reported upward as
    /// `kInternal` tells an operator to look for a logic bug in code that is fine, when the
    /// actual instruction is to re-index a damaged segment. The code is the part callers
    /// branch on; the message is for whoever reads the log.
    ///
    /// An Ok status is returned unchanged — annotating a success is meaningless, and
    /// building a failure out of one would invent an error that did not happen.
    Status WithContext(std::string_view where) const {
        if (ok()) return *this;
        return {code_, std::string(where) + ": " + message_};
    }

private:
    Status(Code c, std::string m) : code_(c), message_(std::move(m)) {}

    Code code_ = Code::kOk;
    std::string message_;
};

// Result<T> is a value or the reason there isn't one.
template <typename T>
class Result {
public:
    Result(T value) : payload_(std::move(value)) {}            // NOLINT: implicit is the point
    Result(Status status) : payload_(std::move(status)) {}      // NOLINT

    bool ok() const { return std::holds_alternative<T>(payload_); }

    const Status& status() const { return std::get<Status>(payload_); }

    T& value() { return std::get<T>(payload_); }
    const T& value() const { return std::get<T>(payload_); }

    T&& take() { return std::move(std::get<T>(payload_)); }

private:
    std::variant<T, Status> payload_;
};

// MERIDIAN_TRY propagates a failure out of the current function, so a long chain
// of fallible calls reads as a sequence of statements rather than nested ifs.
#define MERIDIAN_TRY(expr)                        \
    do {                                        \
        ::meridian::Status _s = (expr);           \
        if (!_s.ok()) return _s;                \
    } while (false)

// MERIDIAN_ASSIGN binds a Result's value or returns its Status.
//
// The temporary is named with __COUNTER__ rather than __LINE__. Two levels of
// indirection are needed either way, because `_res_##__LINE__` pastes the literal
// text "__LINE__" instead of the number — but __LINE__ still collides when two
// uses appear on one line, which is legal code that should not fail to compile.
// __COUNTER__ is unique per expansion. It is a compiler extension rather than
// standard C++, and is supported by clang, gcc, and msvc.
#define MERIDIAN_CONCAT_INNER(a, b) a##b
#define MERIDIAN_CONCAT(a, b) MERIDIAN_CONCAT_INNER(a, b)

#define MERIDIAN_ASSIGN_IMPL(lhs, expr, tmp)      \
    auto tmp = (expr);                          \
    if (!tmp.ok()) return tmp.status();         \
    lhs = tmp.take()

#define MERIDIAN_ASSIGN(lhs, expr) \
    MERIDIAN_ASSIGN_IMPL(lhs, expr, MERIDIAN_CONCAT(_quarry_res_, __COUNTER__))

}  // namespace meridian
