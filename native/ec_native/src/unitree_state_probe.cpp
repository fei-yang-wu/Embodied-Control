#include "unitree_state_probe.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <thread>

#include <unitree/dds_wrapper/common/crc.h>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace ec_native {
namespace {

using LowState = unitree_hg::msg::dds_::LowState_;
using unitree::robot::ChannelFactory;
using unitree::robot::ChannelSubscriber;

std::uint64_t monotonic_ns() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void update_max(std::atomic<std::uint64_t>& destination,
                std::uint64_t value) noexcept {
  auto current = destination.load(std::memory_order_relaxed);
  while (current < value &&
         !destination.compare_exchange_weak(current, value,
                                            std::memory_order_relaxed)) {
  }
}

}  // namespace

struct UnitreeStateProbe::Impl {
  std::shared_ptr<ChannelSubscriber<LowState>> subscriber;
};

struct UnitreeStateProbe::Slot {
  std::atomic<std::uint64_t> sequence{0};
  std::array<std::atomic<float>, 29> joint_position{};
  std::array<std::atomic<float>, 29> joint_velocity{};
  std::array<std::atomic<float>, 29> joint_torque{};
  std::array<std::atomic<std::uint32_t>, 29> motor_state{};
  std::array<std::atomic<float>, 4> quaternion{};
  std::array<std::atomic<float>, 3> gyroscope{};
  std::array<std::atomic<float>, 3> accelerometer{};
  std::atomic<std::uint32_t> mode_machine{0};
};

UnitreeStateProbe::UnitreeStateProbe(const std::string& network_interface,
                                     int dds_domain, std::size_t sample_capacity)
    : impl_(std::make_unique<Impl>()),
      slot_(std::make_unique<Slot>()),
      sample_capacity_(sample_capacity),
      sample_values_(sample_capacity * kProbeSampleWidth, 0.0F),
      sample_times_ns_(sample_capacity, 0),
      sample_ticks_(sample_capacity, 0) {
  ChannelFactory::Instance()->Init(dds_domain, network_interface);
  impl_->subscriber =
      std::make_shared<ChannelSubscriber<LowState>>("rt/lowstate");
  impl_->subscriber->InitChannel(
      std::bind(&UnitreeStateProbe::low_state_handler, this,
                std::placeholders::_1),
      1);
}

UnitreeStateProbe::~UnitreeStateProbe() { impl_->subscriber.reset(); }

void UnitreeStateProbe::low_state_handler(const void* message) noexcept {
  const auto& state = *static_cast<const LowState*>(message);
  if (state.crc() !=
      crc32_core(
          reinterpret_cast<std::uint32_t*>(const_cast<LowState*>(&state)),
          (sizeof(LowState) >> 2) - 1)) {
    crc_errors_.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  const auto receive_ns = monotonic_ns();
  const auto previous_receive_ns =
      last_receive_ns_.exchange(receive_ns, std::memory_order_relaxed);
  std::uint64_t empty_timestamp = 0;
  first_receive_ns_.compare_exchange_strong(empty_timestamp, receive_ns);
  if (previous_receive_ns != 0 && receive_ns > previous_receive_ns) {
    update_max(max_gap_ns_, receive_ns - previous_receive_ns);
  }

  const auto tick = state.tick();
  const auto previous_tick = last_tick_.exchange(tick);
  if (tick_seen_.exchange(true)) {
    if (tick == previous_tick) {
      duplicate_ticks_.fetch_add(1, std::memory_order_relaxed);
    }
  } else {
    first_tick_.store(tick, std::memory_order_relaxed);
  }

  const auto sequence = slot_->sequence.load(std::memory_order_relaxed);
  slot_->sequence.store(sequence + 1, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);

  bool finite = true;
  std::uint32_t motor_errors = 0;
  for (std::size_t index = 0; index < 29; ++index) {
    const auto& motor = state.motor_state()[index];
    slot_->joint_position[index].store(motor.q(), std::memory_order_relaxed);
    slot_->joint_velocity[index].store(motor.dq(), std::memory_order_relaxed);
    slot_->joint_torque[index].store(motor.tau_est(),
                                     std::memory_order_relaxed);
    slot_->motor_state[index].store(motor.motorstate(),
                                    std::memory_order_relaxed);
    finite = finite && std::isfinite(motor.q()) &&
             std::isfinite(motor.dq()) && std::isfinite(motor.tau_est());
    if (motor.motorstate() != 0) {
      motor_errors |= 1U << index;
      std::uint32_t expected = 0;
      first_motor_error_code_[index].compare_exchange_strong(
          expected, motor.motorstate(), std::memory_order_relaxed);
    }
  }

  const auto& imu = state.imu_state();
  float quaternion_norm_squared = 0.0F;
  for (std::size_t index = 0; index < 4; ++index) {
    const float value = imu.quaternion()[index];
    slot_->quaternion[index].store(value, std::memory_order_relaxed);
    finite = finite && std::isfinite(value);
    quaternion_norm_squared += value * value;
  }
  finite = finite && quaternion_norm_squared >= 0.5F &&
           quaternion_norm_squared <= 1.5F;
  for (std::size_t index = 0; index < 3; ++index) {
    const float gyro = imu.gyroscope()[index];
    const float accel = imu.accelerometer()[index];
    slot_->gyroscope[index].store(gyro, std::memory_order_relaxed);
    slot_->accelerometer[index].store(accel, std::memory_order_relaxed);
    finite = finite && std::isfinite(gyro) && std::isfinite(accel);
  }

  slot_->mode_machine.store(state.mode_machine(), std::memory_order_relaxed);
  slot_->sequence.store(sequence + 2, std::memory_order_release);

  const auto captured = captured_.load(std::memory_order_relaxed);
  if (captured < sample_capacity_) {
    float* row = sample_values_.data() + captured * kProbeSampleWidth;
    for (std::size_t index = 0; index < 29; ++index) {
      const auto& motor = state.motor_state()[index];
      row[index] = motor.q();
      row[29 + index] = motor.dq();
      row[58 + index] = motor.tau_est();
    }
    for (std::size_t index = 0; index < 4; ++index) {
      row[87 + index] = imu.quaternion()[index];
    }
    for (std::size_t index = 0; index < 3; ++index) {
      row[91 + index] = imu.gyroscope()[index];
      row[94 + index] = imu.accelerometer()[index];
    }
    sample_times_ns_[captured] = receive_ns;
    sample_ticks_[captured] = tick;
    captured_.store(captured + 1, std::memory_order_release);
  }

  if (!finite) {
    nonfinite_samples_.fetch_add(1, std::memory_order_relaxed);
  }
  if (motor_errors != 0) {
    motor_error_samples_.fetch_add(1, std::memory_order_relaxed);
    motor_error_mask_.fetch_or(motor_errors, std::memory_order_relaxed);
  }
  samples_.fetch_add(1, std::memory_order_release);
}

std::size_t UnitreeStateProbe::captured_samples() const noexcept {
  return static_cast<std::size_t>(captured_.load(std::memory_order_acquire));
}

bool UnitreeStateProbe::wait_for_samples(std::uint64_t count,
                                         double timeout_seconds) const {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(timeout_seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    if (samples_.load(std::memory_order_acquire) >= count) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return samples_.load(std::memory_order_acquire) >= count;
}

UnitreeProbeSnapshot UnitreeStateProbe::snapshot() const noexcept {
  UnitreeProbeSnapshot result;
  result.samples = samples_.load(std::memory_order_acquire);
  result.crc_errors = crc_errors_.load(std::memory_order_relaxed);
  result.nonfinite_samples =
      nonfinite_samples_.load(std::memory_order_relaxed);
  result.motor_error_samples =
      motor_error_samples_.load(std::memory_order_relaxed);
  result.duplicate_ticks = duplicate_ticks_.load(std::memory_order_relaxed);
  result.first_receive_ns = first_receive_ns_.load(std::memory_order_relaxed);
  result.last_receive_ns = last_receive_ns_.load(std::memory_order_relaxed);
  result.max_gap_ns = max_gap_ns_.load(std::memory_order_relaxed);
  result.first_tick = first_tick_.load(std::memory_order_relaxed);
  result.last_tick = last_tick_.load(std::memory_order_relaxed);
  result.motor_error_mask = motor_error_mask_.load(std::memory_order_relaxed);
  for (std::size_t index = 0; index < 29; ++index) {
    result.first_motor_error_code[index] =
        first_motor_error_code_[index].load(std::memory_order_relaxed);
  }

  for (int attempt = 0; attempt < 8; ++attempt) {
    const auto first = slot_->sequence.load(std::memory_order_acquire);
    if (first == 0 || (first & 1U) != 0) {
      continue;
    }
    for (std::size_t index = 0; index < 29; ++index) {
      result.joint_position[index] =
          slot_->joint_position[index].load(std::memory_order_relaxed);
      result.joint_velocity[index] =
          slot_->joint_velocity[index].load(std::memory_order_relaxed);
      result.joint_torque[index] =
          slot_->joint_torque[index].load(std::memory_order_relaxed);
      result.motor_state[index] =
          slot_->motor_state[index].load(std::memory_order_relaxed);
    }
    for (std::size_t index = 0; index < 4; ++index) {
      result.quaternion[index] =
          slot_->quaternion[index].load(std::memory_order_relaxed);
    }
    for (std::size_t index = 0; index < 3; ++index) {
      result.gyroscope[index] =
          slot_->gyroscope[index].load(std::memory_order_relaxed);
      result.accelerometer[index] =
          slot_->accelerometer[index].load(std::memory_order_relaxed);
    }
    result.mode_machine =
        slot_->mode_machine.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (first == slot_->sequence.load(std::memory_order_relaxed)) {
      result.snapshot_consistent = true;
      break;
    }
  }
  return result;
}

}  // namespace ec_native
