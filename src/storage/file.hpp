#pragma once

// The file layer, deliberately small: positioned reads, one atomic write, directory
// listing. Everything above it addresses a segment by byte offset, so this is the only
// place that knows a file descriptor exists.

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "meridian/status.hpp"

namespace meridian::storage {

/// A random-access source of bytes.
///
/// The indirection exists so a posting list can be decoded out of a buffer in a test
/// and out of a segment file in production with the same code. A bug in the block
/// framing is far easier to pin down when it can be reproduced without a file.
class ByteSource {
public:
    virtual ~ByteSource() = default;

    /// Reads exactly `length` bytes. A short read is an error, not a partial success:
    /// a truncated block must surface as kCorruption rather than as a decode of
    /// whatever happened to be there.
    virtual Status ReadAt(uint64_t offset, size_t length, uint8_t* dst) const = 0;

    virtual uint64_t size() const = 0;
};

/// A file opened for positioned reads.
///
/// pread, not mmap. The choice is worth stating because mmap is the usual reflex for a
/// read-only index, and three things rule it out here:
///
///  1. `SegmentReader::bytes_read()` has to be honest, and under mmap it cannot be.
///     A page fault is invisible to the process, so a reader that mmaps the whole
///     segment and touches one posting reports either zero bytes or the whole file
///     depending on how you choose to lie. The entire point of the counter is to make
///     the projection and skipping claims measurable, and a measurement that cannot be
///     taken is worse than no counter at all.
///  2. An I/O error on a mapped page is SIGBUS, not a return value. status.hpp exists
///     because a search engine's failures are ordinary control flow; swapping a
///     Status for a fatal signal on the one path that touches real hardware inverts
///     that.
///  3. The corrupt-block test needs to truncate or scribble on a file that a reader
///     already holds. With a mapping that is undefined behaviour; with pread it is
///     just a short read or a failed checksum.
///
/// The cost is a syscall per block read. A block is 128 postings, so the syscall is
/// amortised over 128 decodes, and the skip path reads far fewer blocks than it skips
/// over — which is the thing being optimised, and it is unaffected either way.
class ReadableFile {
public:
    static Result<std::shared_ptr<ReadableFile>> Open(const std::string& path);

    ~ReadableFile();
    ReadableFile(const ReadableFile&) = delete;
    ReadableFile& operator=(const ReadableFile&) = delete;

    Status ReadAt(uint64_t offset, size_t length, uint8_t* dst) const;
    uint64_t size() const { return size_; }
    const std::string& path() const { return path_; }

private:
    ReadableFile(int fd, uint64_t size, std::string path)
        : fd_(fd), size_(size), path_(std::move(path)) {}

    int fd_;
    uint64_t size_;
    std::string path_;
};

/// A ByteSource that adds every byte it returns to a counter it does not own.
///
/// The counter is shared by pointer rather than held by value because several readers
/// of the same segment file exist at once — a deletion publishes a new reader over the
/// same immutable bytes — and each of them has to report its *own* bytes_read. Keeping
/// the counter outside the file object is what lets the file handle be shared while
/// the accounting is not.
class CountingByteSource : public ByteSource {
public:
    CountingByteSource(std::shared_ptr<ReadableFile> file,
                       std::shared_ptr<std::atomic<uint64_t>> counter)
        : file_(std::move(file)), counter_(std::move(counter)) {}

    Status ReadAt(uint64_t offset, size_t length, uint8_t* dst) const override;
    uint64_t size() const override { return file_->size(); }

private:
    std::shared_ptr<ReadableFile> file_;
    std::shared_ptr<std::atomic<uint64_t>> counter_;
};

/// An in-memory ByteSource, for tests and for posting lists that were just built.
class MemoryByteSource : public ByteSource {
public:
    explicit MemoryByteSource(std::vector<uint8_t> bytes) : bytes_(std::move(bytes)) {}

    Status ReadAt(uint64_t offset, size_t length, uint8_t* dst) const override;
    uint64_t size() const override { return bytes_.size(); }

    /// Exposed so a test can flip a bit and watch the checksum catch it.
    std::vector<uint8_t>& mutable_bytes() { return bytes_; }

private:
    std::vector<uint8_t> bytes_;
};

Status EnsureDirectory(const std::string& path);

/// File names only, not full paths; the caller knows the directory it asked about.
Result<std::vector<std::string>> ListDirectory(const std::string& path);

/// Writes `contents` so that a reader sees either the whole file or no file at all.
///
/// temp file -> fsync(temp) -> rename -> fsync(directory). All four steps matter and
/// the last is the one usually skipped: rename is atomic with respect to other
/// processes immediately, but the *directory entry* is not durable until the directory
/// itself is synced, so a crash can leave a correctly-fsynced segment that no longer
/// has a name. That is the failure that turns a published generation into a dangling
/// reference on restart.
Status WriteFileAtomic(const std::string& path, std::span<const uint8_t> contents);

Status RemoveFile(const std::string& path);
bool FileExists(const std::string& path);
Result<uint64_t> FileSize(const std::string& path);

}  // namespace meridian::storage
