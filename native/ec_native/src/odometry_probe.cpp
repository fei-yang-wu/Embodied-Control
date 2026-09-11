#include "odometry_probe.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <thread>

#include <time.h>

#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace ec_native {
namespace {

using OdometryState = unitree_go::msg::dds_::SportModeState_;
using unitree::robot::ChannelFactory;
using unitree::robot::ChannelSubscriber;

std::uint64_t monotonic_ns_probe() noexcept {
  timespec now{};
  clock_gettime(CLOCK_MONOTONIC, &now);
  return static_cast<std::uint64_t>(now.tv_sec) * 1000000000ULL +
         static_cast<std::uint64_t>(now.tv_nsec);
}

}  // namespace

struct OdometryProbe::Impl {
  std::shared_ptr<ChannelSubscriber<OdometryState>> subscriber;
};

struct OdometryProbe::Slot {
  std::atomic<std::uint64_t> sequence{0};
  std::array<std::atomic<float>, 3> first_position{};
  std::array<std::atomic<float>, 3> position{};
  std::array<std::atomic<float>, 3> velocity{};
  std::array<std::atomic<float>, 4> quaternion_wxyz{};
  std::atomic<std::uint32_t> error_code{0};
  std::atomic<std::uint8_t> mode{0};
};

OdometryProbe::OdometryProbe(const std::string& network_interface,
                             const std::string& topic, int dds_domain)
    : impl_(std::make_unique<Impl>()), slot_(std::make_unique<Slot>()) {
  ChannelFactory::Instance()->Init(dds_domain, network_interface);
  impl_->subscriber = std::make_shared<ChannelSubscriber<OdometryState>>(topic);
  impl_->subscriber->InitChannel(
      std::bind(&OdometryProbe::handler, this, std::placeholders::_1), 1);
}

OdometryProbe::~OdometryProbe() { impl_->subscriber.reset(); }

void OdometryProbe::handler(const void* message) noexcept {
  const auto& state = *static_cast<const OdometryState*>(message);
  const auto& position = state.position();
  bool finite = true;
  for (const float value : position) {
    finite = finite && std::isfinite(value);
  }
  if (!finite) {
    nonfinite_frames_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  const std::uint64_t now = monotonic_ns_probe();
  const std::uint64_t previous = last_receive_ns_.load(std::memory_order_relaxed);
  if (previous != 0 && now > previous) {
    const std::uint64_t gap = now - previous;
    std::uint64_t seen = max_gap_ns_.load(std::memory_order_relaxed);
    while (gap > seen && !max_gap_ns_.compare_exchange_weak(seen, gap)) {
    }
  }
  const std::uint64_t sequence = slot_->sequence.load(std::memory_order_relaxed);
  slot_->sequence.store(sequence + 1, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  const bool first = frames_.load(std::memory_order_relaxed) == 0;
  for (std::size_t index = 0; index < 3; ++index) {
    slot_->position[index].store(position[index], std::memory_order_relaxed);
    slot_->velocity[index].store(state.velocity()[index], std::memory_order_relaxed);
    if (first) {
      slot_->first_position[index].store(position[index], std::memory_order_relaxed);
    }
  }
  for (std::size_t index = 0; index < 4; ++index) {
    slot_->quaternion_wxyz[index].store(state.imu_state().quaternion()[index],
                                        std::memory_order_relaxed);
  }
  slot_->error_code.store(state.error_code(), std::memory_order_relaxed);
  slot_->mode.store(state.mode(), std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  slot_->sequence.store(sequence + 2, std::memory_order_relaxed);
  if (first) {
    first_receive_ns_.store(now, std::memory_order_relaxed);
  }
  last_receive_ns_.store(now, std::memory_order_relaxed);
  frames_.fetch_add(1, std::memory_order_release);
}

bool OdometryProbe::wait_for_frames(std::uint64_t count,
                                    double timeout_seconds) const {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(timeout_seconds);
  while (frames_.load(std::memory_order_acquire) < count) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return true;
}

OdometryProbeSnapshot OdometryProbe::snapshot() const noexcept {
  OdometryProbeSnapshot out;
  out.frames = frames_.load(std::memory_order_acquire);
  out.nonfinite_frames = nonfinite_frames_.load(std::memory_order_relaxed);
  out.first_receive_ns = first_receive_ns_.load(std::memory_order_relaxed);
  out.last_receive_ns = last_receive_ns_.load(std::memory_order_relaxed);
  out.max_gap_ns = max_gap_ns_.load(std::memory_order_relaxed);
  for (int attempt = 0; attempt < 8; ++attempt) {
    const std::uint64_t first = slot_->sequence.load(std::memory_order_acquire);
    if (first == 0 || (first & 1ULL) != 0) {
      continue;
    }
    for (std::size_t index = 0; index < 3; ++index) {
      out.first_position[index] = slot_->first_position[index].load(std::memory_order_relaxed);
      out.position[index] = slot_->position[index].load(std::memory_order_relaxed);
      out.velocity[index] = slot_->velocity[index].load(std::memory_order_relaxed);
    }
    for (std::size_t index = 0; index < 4; ++index) {
      out.quaternion_wxyz[index] = slot_->quaternion_wxyz[index].load(std::memory_order_relaxed);
    }
    out.error_code = slot_->error_code.load(std::memory_order_relaxed);
    out.mode = slot_->mode.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (first == slot_->sequence.load(std::memory_order_relaxed)) {
      break;
    }
  }
  return out;
}

}  // namespace ec_native
