#pragma once

// A fixed-size thread pool with a bounded queue.
//
// Two users with the same requirement: the query server needs it to cap how much work
// is in flight, and the pipeline needs it to overlap the lexical and semantic paths.
// Both need the same refusal semantics, which is why this is one type rather than two.
//
// `TryPost` never blocks and never grows the queue. That is the whole point: a pool
// whose queue grows converts overload into latency, and latency is the thing the
// caller is trying to bound. The caller decides what refusal means — the server turns
// it into kOverloaded, the pipeline turns it into "run this stage inline instead" —
// and neither decision belongs in the pool.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace meridian::serve {

class TaskPool {
public:
    using Task = std::function<void()>;

    TaskPool(size_t threads, size_t queue_capacity)
        : queue_capacity_(queue_capacity) {
        workers_.reserve(threads);
        for (size_t i = 0; i < threads; ++i) {
            workers_.emplace_back([this] { Loop(); });
        }
    }

    ~TaskPool() { Shutdown(); }

    TaskPool(const TaskPool&) = delete;
    TaskPool& operator=(const TaskPool&) = delete;

    /// Enqueues work if there is room. Returns false when the queue is full or the
    /// pool is shutting down; the caller must handle that rather than retry in a loop,
    /// because a retry loop is an unbounded queue with extra steps.
    bool TryPost(Task task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) return false;
            if (queue_.size() >= queue_capacity_) {
                ++refused_;
                return false;
            }
            queue_.push(std::move(task));
            ++accepted_;
        }
        cv_.notify_one();
        return true;
    }

    /// Queue depth now. Sampled for metrics; inherently stale by the time it returns.
    size_t pending() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    /// True once Shutdown has begun. Callers need this to tell "no capacity" from "no
    /// longer accepting work": the first is backpressure the client should retry
    /// elsewhere, the second never succeeds and must not look like overload.
    bool stopping() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stopping_;
    }

    size_t threads() const { return workers_.size(); }
    size_t queue_capacity() const { return queue_capacity_; }
    uint64_t accepted() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return accepted_;
    }
    uint64_t refused() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return refused_;
    }

    /// Drains tasks already queued, then joins. Shutdown drains rather than discards
    /// because a submitted query has a caller waiting on a future, and dropping it
    /// would leave that caller blocked forever.
    void Shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) return;
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
        workers_.clear();
    }

private:
    void Loop() {
        for (;;) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
                if (queue_.empty()) {
                    if (stopping_) return;
                    continue;
                }
                task = std::move(queue_.front());
                queue_.pop();
            }
            task();
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<Task> queue_;
    std::vector<std::thread> workers_;
    const size_t queue_capacity_;
    bool stopping_ = false;
    uint64_t accepted_ = 0;
    uint64_t refused_ = 0;
};

}  // namespace meridian::serve
