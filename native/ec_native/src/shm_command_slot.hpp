// Single-slot latest-wins command buffer over POSIX shared memory.
//
// One writer process, one reader process, no broker. A seqlock guards the
// slot: the writer bumps the sequence word to odd, writes, bumps to even; the
// reader retries until it sees a stable even sequence word. Stamps use
// CLOCK_MONOTONIC, which is one system-wide clock on Linux, so the Python
// side can age packets against time.monotonic().

#pragma once

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

namespace ec_native {

inline double monotonic_now() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

// A stride-5 reference block can carry 55 x 62 = 3,410 values, and a latent
// plan carries slots x z_dim (30 x 256 = 7,680 for the hold-1 tracker). The
// extra capacity keeps transport renewal outside the 50 Hz encoder cadence.
constexpr std::uint32_t kMaxValues = 16384;
constexpr std::uint64_t kMagic = 0x45434e4154495633ull;  // "ECNATIV3"
constexpr int kSnapshotAttempts = 8;

struct Slot {
  std::uint64_t magic;
  std::uint64_t seq_word;   // seqlock word: odd while the writer is inside
  std::uint64_t sequence;   // publisher-lifetime monotonic packet sequence
  double recv_stamp;        // CLOCK_MONOTONIC at publish
  double sender_stamp;      // diagnostic only
  std::uint32_t interface_tag;  // 0 explicit, 1 latent, 2 chunk, 10 request
  std::uint32_t length;
  float values[kMaxValues];
};

static_assert(std::atomic_ref<std::uint64_t>::is_always_lock_free);
static_assert(std::atomic_ref<std::uint32_t>::is_always_lock_free);
static_assert(std::atomic_ref<double>::is_always_lock_free);
static_assert(std::atomic_ref<float>::is_always_lock_free);

class ShmSlot {
 public:
  ShmSlot(const std::string& name, bool create) : name_(name), owner_(create) {
    const int flags = create ? (O_CREAT | O_EXCL | O_RDWR) : O_RDWR;
    fd_ = ::shm_open(name.c_str(), flags, 0600);
    if (fd_ < 0) {
      const std::string detail =
          create && errno == EEXIST
              ? "already exists; connect to it or remove the stale segment"
              : std::string(std::strerror(errno));
      throw std::runtime_error("shm_open failed for " + name + ": " +
                               detail);
    }
    if (create && ::ftruncate(fd_, sizeof(Slot)) != 0) {
      ::close(fd_);
      ::shm_unlink(name_.c_str());
      throw std::runtime_error("ftruncate failed for " + name);
    }
    void* mem = ::mmap(nullptr, sizeof(Slot), PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd_, 0);
    if (mem == MAP_FAILED) {
      ::close(fd_);
      if (create) {
        ::shm_unlink(name_.c_str());
      }
      throw std::runtime_error("mmap failed for " + name);
    }
    slot_ = static_cast<Slot*>(mem);
    if (create) {
      std::memset(slot_, 0, sizeof(Slot));
      std::atomic_ref<std::uint64_t>(slot_->magic)
          .store(kMagic, std::memory_order_release);
    } else if (std::atomic_ref<std::uint64_t>(slot_->magic)
                   .load(std::memory_order_acquire) != kMagic) {
      unmap();
      throw std::runtime_error("shm segment " + name + " is not an ec_native slot");
    }
  }

  ~ShmSlot() {
    unmap();
    if (owner_) {
      ::shm_unlink(name_.c_str());
    }
  }

  ShmSlot(const ShmSlot&) = delete;
  ShmSlot& operator=(const ShmSlot&) = delete;

  void publish(std::uint64_t sequence, std::uint32_t interface_tag,
               const float* values, std::uint32_t length, double sender_stamp) {
    if (length > kMaxValues) {
      throw std::runtime_error("payload exceeds kMaxValues");
    }
    std::atomic_ref<std::uint64_t> seq(slot_->seq_word);
    const std::uint64_t begin = seq.load(std::memory_order_relaxed);
    seq.store(begin + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    std::atomic_ref<std::uint64_t>(slot_->sequence)
        .store(sequence, std::memory_order_relaxed);
    std::atomic_ref<double>(slot_->recv_stamp)
        .store(monotonic_now(), std::memory_order_relaxed);
    std::atomic_ref<double>(slot_->sender_stamp)
        .store(sender_stamp, std::memory_order_relaxed);
    std::atomic_ref<std::uint32_t>(slot_->interface_tag)
        .store(interface_tag, std::memory_order_relaxed);
    std::atomic_ref<std::uint32_t>(slot_->length)
        .store(length, std::memory_order_relaxed);
    for (std::uint32_t index = 0; index < length; ++index) {
      std::atomic_ref<float>(slot_->values[index])
          .store(values[index], std::memory_order_relaxed);
    }
    std::atomic_thread_fence(std::memory_order_release);
    seq.store(begin + 2, std::memory_order_relaxed);
  }

  // Copies a consistent view into the caller's buffers. Returns false when no
  // packet was ever published.
  bool snapshot(std::uint64_t& sequence, std::uint32_t& interface_tag,
                std::uint32_t& length, double& recv_stamp, double& sender_stamp,
                float* out, std::uint32_t out_capacity,
                std::uint64_t after_sequence = 0) const {
    std::atomic_ref<std::uint64_t> seq(slot_->seq_word);
    for (int attempt = 0; attempt < kSnapshotAttempts; ++attempt) {
      const std::uint64_t s1 = seq.load(std::memory_order_acquire);
      if (s1 == 0) {
        return false;
      }
      if (s1 & 1ull) {
        continue;
      }
      const std::uint64_t got_sequence =
          std::atomic_ref<std::uint64_t>(slot_->sequence)
              .load(std::memory_order_relaxed);
      if (got_sequence <= after_sequence) {
        std::atomic_thread_fence(std::memory_order_acquire);
        if (s1 == seq.load(std::memory_order_relaxed)) {
          return false;
        }
        continue;
      }
      const double got_recv = std::atomic_ref<double>(slot_->recv_stamp)
                                  .load(std::memory_order_relaxed);
      const double got_sender = std::atomic_ref<double>(slot_->sender_stamp)
                                    .load(std::memory_order_relaxed);
      const std::uint32_t got_tag =
          std::atomic_ref<std::uint32_t>(slot_->interface_tag)
              .load(std::memory_order_relaxed);
      const std::uint32_t got_length =
          std::atomic_ref<std::uint32_t>(slot_->length)
              .load(std::memory_order_relaxed);
      if (got_length > kMaxValues || got_length > out_capacity) {
        throw std::runtime_error("snapshot buffer too small");
      }
      for (std::uint32_t index = 0; index < got_length; ++index) {
        out[index] = std::atomic_ref<float>(slot_->values[index])
                         .load(std::memory_order_relaxed);
      }
      std::atomic_thread_fence(std::memory_order_acquire);
      const std::uint64_t s2 = seq.load(std::memory_order_relaxed);
      if (s1 == s2) {
        sequence = got_sequence;
        interface_tag = got_tag;
        length = got_length;
        recv_stamp = got_recv;
        sender_stamp = got_sender;
        return true;
      }
    }
    return false;
  }

 private:
  void unmap() {
    if (slot_ != nullptr) {
      ::munmap(slot_, sizeof(Slot));
      slot_ = nullptr;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  std::string name_;
  bool owner_;
  int fd_ = -1;
  Slot* slot_ = nullptr;
};

}  // namespace ec_native
