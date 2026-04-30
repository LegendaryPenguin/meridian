#include "file.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <sstream>

namespace meridian::storage {
namespace {

std::string Errno(const std::string& what, const std::string& path) {
    std::ostringstream os;
    os << what << " " << path << ": " << std::strerror(errno);
    return os.str();
}

std::string ParentDirectory(const std::string& path) {
    const std::filesystem::path p(path);
    const auto parent = p.parent_path();
    return parent.empty() ? std::string(".") : parent.string();
}

/// fsync on the directory containing `path`, so the rename that created the name is
/// durable and not merely visible.
Status SyncDirectory(const std::string& path) {
    const std::string dir = ParentDirectory(path);
    const int fd = ::open(dir.c_str(), O_RDONLY);
    if (fd < 0) return Status::IoError(Errno("open directory", dir));
    const int rc = ::fsync(fd);
    const int saved = errno;
    ::close(fd);
    if (rc != 0) {
        errno = saved;
        return Status::IoError(Errno("fsync directory", dir));
    }
    return Status::Ok();
}

}  // namespace

Result<std::shared_ptr<ReadableFile>> ReadableFile::Open(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return Status::IoError(Errno("open", path));

    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        const int saved = errno;
        ::close(fd);
        errno = saved;
        return Status::IoError(Errno("fstat", path));
    }

    // make_shared cannot reach the private constructor, and a public one would let a
    // caller build a ReadableFile around an arbitrary descriptor.
    return std::shared_ptr<ReadableFile>(
        new ReadableFile(fd, static_cast<uint64_t>(st.st_size), path));
}

ReadableFile::~ReadableFile() {
    if (fd_ >= 0) ::close(fd_);
}

Status ReadableFile::ReadAt(uint64_t offset, size_t length, uint8_t* dst) const {
    if (length == 0) return Status::Ok();
    if (offset > size_ || length > size_ - offset) {
        std::ostringstream os;
        os << "read of " << length << " bytes at " << offset << " is past the end of " << path_
           << " (" << size_ << " bytes)";
        return Status::Corruption(os.str());
    }

    size_t done = 0;
    while (done < length) {
        // pread is looped rather than called once because a short read is legal and
        // says nothing about the file; only a zero return means end of data.
        const ssize_t n = ::pread(fd_, dst + done, length - done,
                                  static_cast<off_t>(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return Status::IoError(Errno("pread", path_));
        }
        if (n == 0) {
            std::ostringstream os;
            os << "short read on " << path_ << ": got " << done << " of " << length << " bytes";
            return Status::Corruption(os.str());
        }
        done += static_cast<size_t>(n);
    }
    return Status::Ok();
}

Status CountingByteSource::ReadAt(uint64_t offset, size_t length, uint8_t* dst) const {
    MERIDIAN_TRY(file_->ReadAt(offset, length, dst));
    // Counted after the read succeeds: a failed read moved no bytes, and inflating the
    // counter on failure would make a corruption test look like a projection win.
    counter_->fetch_add(length, std::memory_order_relaxed);
    return Status::Ok();
}

Status MemoryByteSource::ReadAt(uint64_t offset, size_t length, uint8_t* dst) const {
    if (offset > bytes_.size() || length > bytes_.size() - offset) {
        std::ostringstream os;
        os << "read of " << length << " bytes at " << offset << " is past the end of a "
           << bytes_.size() << "-byte buffer";
        return Status::Corruption(os.str());
    }
    if (length > 0) std::memcpy(dst, bytes_.data() + offset, length);
    return Status::Ok();
}

Status EnsureDirectory(const std::string& path) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec && !std::filesystem::is_directory(path)) {
        return Status::IoError("create directory " + path + ": " + ec.message());
    }
    return Status::Ok();
}

Result<std::vector<std::string>> ListDirectory(const std::string& path) {
    std::error_code ec;
    std::filesystem::directory_iterator it(path, ec);
    if (ec) return Status::IoError("list directory " + path + ": " + ec.message());

    std::vector<std::string> names;
    for (const auto& entry : it) {
        names.push_back(entry.path().filename().string());
    }
    return names;
}

Status WriteFileAtomic(const std::string& path, std::span<const uint8_t> contents) {
    const std::string temp = path + ".tmp";

    const int fd = ::open(temp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return Status::IoError(Errno("open", temp));

    size_t done = 0;
    while (done < contents.size()) {
        const ssize_t n = ::write(fd, contents.data() + done, contents.size() - done);
        if (n < 0) {
            if (errno == EINTR) continue;
            const int saved = errno;
            ::close(fd);
            ::unlink(temp.c_str());
            errno = saved;
            return Status::IoError(Errno("write", temp));
        }
        done += static_cast<size_t>(n);
    }

    if (::fsync(fd) != 0) {
        const int saved = errno;
        ::close(fd);
        ::unlink(temp.c_str());
        errno = saved;
        return Status::IoError(Errno("fsync", temp));
    }
    if (::close(fd) != 0) {
        ::unlink(temp.c_str());
        return Status::IoError(Errno("close", temp));
    }

    if (::rename(temp.c_str(), path.c_str()) != 0) {
        const int saved = errno;
        ::unlink(temp.c_str());
        errno = saved;
        return Status::IoError(Errno("rename to", path));
    }

    return SyncDirectory(path);
}

Status RemoveFile(const std::string& path) {
    if (::unlink(path.c_str()) != 0) {
        if (errno == ENOENT) return Status::Ok();
        return Status::IoError(Errno("unlink", path));
    }
    return Status::Ok();
}

bool FileExists(const std::string& path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0;
}

Result<uint64_t> FileSize(const std::string& path) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) return Status::IoError(Errno("stat", path));
    return static_cast<uint64_t>(st.st_size);
}

}  // namespace meridian::storage
