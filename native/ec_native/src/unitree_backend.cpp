#include "unitree_backend.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>

#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>

#include <unitree/dds_wrapper/common/crc.h>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

namespace ec_native {
namespace {

using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;
using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelPublisherPtr;
using unitree::robot::ChannelSubscriber;
using unitree::robot::ChannelSubscriberPtr;

constexpr std::uint64_t kWriterPeriodNs = 2000000ULL;
constexpr float kJointLimitFaultMargin = 0.1F;
constexpr int kLocalSnapshotAttempts = 8;

std::uint64_t monotonic_ns_unitree() noexcept {
  timespec now{};
  clock_gettime(CLOCK_MONOTONIC, &now);
  return static_cast<std::uint64_t>(now.tv_sec) * 1000000000ULL +
         static_cast<std::uint64_t>(now.tv_nsec);
}

timespec unitree_timespec(std::uint64_t nanoseconds) noexcept {
  return timespec{
      static_cast<time_t>(nanoseconds / 1000000000ULL),
      static_cast<long>(nanoseconds % 1000000000ULL),
  };
}

void update_unitree_max(std::atomic<std::uint64_t>& destination,
                        std::uint64_t value) noexcept {
  std::uint64_t previous = destination.load(std::memory_order_relaxed);
  while (previous < value &&
         !destination.compare_exchange_weak(previous, value,
                                            std::memory_order_relaxed)) {
  }
}

bool finite_values(std::span<const float> values) {
  return std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); });
}

}  // namespace

struct NativeUnitreeBackend::StateSlot {
  std::atomic<std::uint64_t> sequence{0};
  std::array<std::atomic<float>, kJointCount> joint_position{};
  std::array<std::atomic<float>, kJointCount> joint_velocity{};
  std::array<std::atomic<float>, 4> quaternion{};
  std::array<std::atomic<float>, 3> gyroscope{};
  std::atomic<std::uint64_t> receive_ns{0};
  std::atomic<std::uint8_t> mode_machine{0};
  std::atomic<bool> valid{false};
  std::atomic<bool> faulted{false};
};

struct NativeUnitreeBackend::StateSnapshot {
  std::array<float, kJointCount> joint_position{};
  std::array<float, kJointCount> joint_velocity{};
  std::array<float, 4> quaternion{};
  std::array<float, 3> gyroscope{};
  std::uint64_t receive_ns = 0;
  std::uint8_t mode_machine = 0;
  bool valid = false;
  bool faulted = false;
};

struct NativeUnitreeBackend::CommandSlot {
  std::atomic<std::uint64_t> sequence{0};
  std::array<std::atomic<float>, kJointCount> joint_target{};
  std::atomic<std::uint64_t> receive_ns{0};
};

struct NativeUnitreeBackend::CommandSnapshot {
  std::array<float, kJointCount> joint_target{};
  std::uint64_t receive_ns = 0;
};

struct NativeUnitreeBackend::Impl {
  ChannelPublisherPtr<LowCmd> publisher;
  ChannelSubscriberPtr<LowState> subscriber;
  std::unique_ptr<unitree::robot::b2::MotionSwitcherClient> motion_switcher;
};

NativeUnitreeBackend::NativeUnitreeBackend(
    const std::string& network_interface,
    std::span<const std::size_t> isaac_to_sdk,
    std::span<const float> default_joint_position,
    std::span<const float> stiffness, std::span<const float> damping,
    std::span<const float> joint_lower, std::span<const float> joint_upper,
    bool writes_enabled, int writer_cpu, int writer_fifo_priority,
    bool lock_memory, bool require_realtime, double state_absent_ms,
    double command_stale_ms)
    : impl_(std::make_unique<Impl>()),
      state_slot_(std::make_unique<StateSlot>()),
      command_slot_(std::make_unique<CommandSlot>()),
      writes_enabled_(writes_enabled),
      writer_cpu_(writer_cpu),
      writer_fifo_priority_(writer_fifo_priority),
      lock_memory_(lock_memory),
      require_realtime_(require_realtime),
      state_absent_ms_(state_absent_ms),
      command_stale_ms_(command_stale_ms) {
  if (network_interface.empty() || isaac_to_sdk.size() != kJointCount ||
      default_joint_position.size() != kJointCount ||
      stiffness.size() != kJointCount || damping.size() != kJointCount ||
      joint_lower.size() != kJointCount || joint_upper.size() != kJointCount ||
      !finite_values(default_joint_position) || !finite_values(stiffness) ||
      !finite_values(damping) || !finite_values(joint_lower) ||
      !finite_values(joint_upper) ||
      std::any_of(stiffness.begin(), stiffness.end(),
                  [](float value) { return value < 0.0F; }) ||
      std::any_of(damping.begin(), damping.end(),
                  [](float value) { return value < 0.0F; }) ||
      writer_cpu_ < -1 || writer_cpu_ >= CPU_SETSIZE ||
      writer_fifo_priority_ < 0 ||
      writer_fifo_priority_ > sched_get_priority_max(SCHED_FIFO) ||
      state_absent_ms_ <= 0.0 || command_stale_ms_ <= 0.0) {
    throw std::runtime_error("invalid Unitree DDS backend configuration");
  }
  std::array<bool, kJointCount> seen{};
  for (std::size_t index = 0; index < kJointCount; ++index) {
    if (isaac_to_sdk[index] >= kJointCount || seen[isaac_to_sdk[index]]) {
      throw std::runtime_error("isaac_to_sdk must be a permutation of 0..28");
    }
    seen[isaac_to_sdk[index]] = true;
    isaac_to_sdk_[index] = isaac_to_sdk[index];
  }
  std::copy(default_joint_position.begin(), default_joint_position.end(),
            default_joint_position_.begin());
  std::copy(stiffness.begin(), stiffness.end(), stiffness_.begin());
  std::copy(damping.begin(), damping.end(), damping_.begin());
  for (std::size_t index = 0; index < kJointCount; ++index) {
    if (joint_lower[index] > joint_upper[index]) {
      throw std::runtime_error("Unitree joint lower limit exceeds upper limit");
    }
    joint_lower_[index] = joint_lower[index];
    joint_upper_[index] = joint_upper[index];
  }
  state_slot_->quaternion[0].store(1.0F, std::memory_order_relaxed);
  for (std::size_t index = 0; index < kJointCount; ++index) {
    command_slot_->joint_target[index].store(default_joint_position_[index],
                                              std::memory_order_relaxed);
  }
  state_cache_.projected_gravity = {0.0F, 0.0F, -1.0F};

  ChannelFactory::Instance()->Init(0, network_interface);
  impl_->publisher =
      std::make_shared<ChannelPublisher<LowCmd>>("rt/lowcmd");
  impl_->publisher->InitChannel();
  impl_->subscriber =
      std::make_shared<ChannelSubscriber<LowState>>("rt/lowstate");
  impl_->subscriber->InitChannel(
      std::bind(&NativeUnitreeBackend::low_state_handler, this,
                std::placeholders::_1),
      1);
  writer_thread_ = std::thread(&NativeUnitreeBackend::writer_loop, this);
}

NativeUnitreeBackend::~NativeUnitreeBackend() {
  force_damp();
  if (writes_enabled_) {
    timespec short_pause{0, 20000000};
    nanosleep(&short_pause, nullptr);
  }
  stop_requested_.store(true);
  if (writer_thread_.joinable()) {
    writer_thread_.join();
  }
  impl_->subscriber.reset();
}

void NativeUnitreeBackend::reset() {}

bool NativeUnitreeBackend::snapshot_state(StateSnapshot& destination) const
    noexcept {
  for (int attempt = 0; attempt < kLocalSnapshotAttempts; ++attempt) {
    const std::uint64_t first =
        state_slot_->sequence.load(std::memory_order_acquire);
    if (first == 0 || (first & 1ULL) != 0) {
      continue;
    }
    for (std::size_t index = 0; index < kJointCount; ++index) {
      destination.joint_position[index] =
          state_slot_->joint_position[index].load(std::memory_order_relaxed);
      destination.joint_velocity[index] =
          state_slot_->joint_velocity[index].load(std::memory_order_relaxed);
    }
    for (std::size_t index = 0; index < destination.quaternion.size(); ++index) {
      destination.quaternion[index] =
          state_slot_->quaternion[index].load(std::memory_order_relaxed);
    }
    for (std::size_t index = 0; index < destination.gyroscope.size(); ++index) {
      destination.gyroscope[index] =
          state_slot_->gyroscope[index].load(std::memory_order_relaxed);
    }
    destination.receive_ns =
        state_slot_->receive_ns.load(std::memory_order_relaxed);
    destination.mode_machine =
        state_slot_->mode_machine.load(std::memory_order_relaxed);
    destination.valid = state_slot_->valid.load(std::memory_order_relaxed);
    destination.faulted =
        state_slot_->faulted.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (first == state_slot_->sequence.load(std::memory_order_relaxed)) {
      return destination.valid;
    }
  }
  return false;
}

bool NativeUnitreeBackend::snapshot_command(CommandSnapshot& destination) const
    noexcept {
  for (int attempt = 0; attempt < kLocalSnapshotAttempts; ++attempt) {
    const std::uint64_t first =
        command_slot_->sequence.load(std::memory_order_acquire);
    if ((first & 1ULL) != 0) {
      continue;
    }
    for (std::size_t index = 0; index < kJointCount; ++index) {
      destination.joint_target[index] =
          command_slot_->joint_target[index].load(std::memory_order_relaxed);
    }
    destination.receive_ns =
        command_slot_->receive_ns.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (first == command_slot_->sequence.load(std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

void NativeUnitreeBackend::low_state_handler(const void* message) noexcept {
  const LowState& input = *static_cast<const LowState*>(message);
  if (input.crc() != crc32_core(
                         reinterpret_cast<std::uint32_t*>(
                             const_cast<LowState*>(&input)),
                         (sizeof(LowState) >> 2) - 1)) {
    crc_errors_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  std::uint64_t sequence =
      state_slot_->sequence.load(std::memory_order_relaxed);
  state_slot_->sequence.store(sequence + 1, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  bool faulted = state_slot_->faulted.load(std::memory_order_relaxed);
  const bool was_faulted = faulted;
  for (std::size_t isaac = 0; isaac < kJointCount; ++isaac) {
    const std::size_t sdk = isaac_to_sdk_[isaac];
    const auto& motor = input.motor_state()[sdk];
    state_slot_->joint_position[isaac].store(motor.q(),
                                              std::memory_order_relaxed);
    state_slot_->joint_velocity[isaac].store(motor.dq(),
                                              std::memory_order_relaxed);
    const bool invalid = !std::isfinite(motor.q()) ||
                         !std::isfinite(motor.dq()) ||
                         motor.q() <
                             joint_lower_[isaac] - kJointLimitFaultMargin ||
                         motor.q() >
                             joint_upper_[isaac] + kJointLimitFaultMargin ||
                         std::abs(motor.dq()) > 35.0F ||
                         motor.motorstate() != 0 ||
                         std::max(motor.temperature()[0],
                                  motor.temperature()[1]) >= 90;
    faulted = faulted || invalid;
  }
  const auto& input_quaternion = input.imu_state().quaternion();
  const auto& input_gyroscope = input.imu_state().gyroscope();
  float quaternion_norm_squared = 0.0F;
  for (std::size_t index = 0; index < 4; ++index) {
    const float value = input_quaternion[index];
    state_slot_->quaternion[index].store(value, std::memory_order_relaxed);
    quaternion_norm_squared += value * value;
    faulted = faulted || !std::isfinite(value);
  }
  for (std::size_t index = 0; index < 3; ++index) {
    const float value = input_gyroscope[index];
    state_slot_->gyroscope[index].store(value, std::memory_order_relaxed);
    faulted = faulted || !std::isfinite(value);
  }
  faulted = faulted || quaternion_norm_squared < 0.5F ||
            quaternion_norm_squared > 1.5F;
  state_slot_->receive_ns.store(monotonic_ns_unitree(),
                                std::memory_order_relaxed);
  state_slot_->mode_machine.store(input.mode_machine(),
                                  std::memory_order_relaxed);
  state_slot_->valid.store(true, std::memory_order_relaxed);
  state_slot_->faulted.store(faulted, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  state_slot_->sequence.store(sequence + 2, std::memory_order_relaxed);
  if (faulted) {
    hardware_fault_latched_.store(true, std::memory_order_release);
    if (!was_faulted) {
      hardware_faults_.fetch_add(1, std::memory_order_relaxed);
    }
    mode_.store(UnitreeMode::kDamp);
  }
}

const RobotState& NativeUnitreeBackend::read_state() noexcept {
  StateSnapshot snapshot;
  if (!snapshot_state(snapshot)) {
    return state_cache_;
  }
  state_cache_.joint_position = snapshot.joint_position;
  state_cache_.joint_velocity = snapshot.joint_velocity;
  state_cache_.base_angular_velocity = snapshot.gyroscope;
  const float w = snapshot.quaternion[0];
  const float x = snapshot.quaternion[1];
  const float y = snapshot.quaternion[2];
  const float z = snapshot.quaternion[3];
  state_cache_.projected_gravity = {
      2.0F * (x * z - w * y),
      2.0F * (y * z + w * x),
      -(1.0F - 2.0F * (x * x + y * y)),
  };
  return state_cache_;
}

void NativeUnitreeBackend::write_target(
    std::span<const float> target) noexcept {
  if (target.size() != kJointCount ||
      !std::all_of(target.begin(), target.end(),
                   [](float value) { return std::isfinite(value); })) {
    force_damp();
    return;
  }
  const std::uint64_t sequence =
      command_slot_->sequence.load(std::memory_order_relaxed);
  command_slot_->sequence.store(sequence + 1, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  for (std::size_t index = 0; index < kJointCount; ++index) {
    command_slot_->joint_target[index].store(target[index],
                                              std::memory_order_relaxed);
  }
  command_slot_->receive_ns.store(monotonic_ns_unitree(),
                                  std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  command_slot_->sequence.store(sequence + 2, std::memory_order_relaxed);
}

void NativeUnitreeBackend::damp() noexcept {
  const UnitreeMode mode = mode_.load();
  if (mode == UnitreeMode::kWait || mode == UnitreeMode::kControl) {
    mode_.store(UnitreeMode::kDamp);
  }
}

void NativeUnitreeBackend::force_damp() noexcept {
  mode_.store(UnitreeMode::kDamp);
}

bool NativeUnitreeBackend::state_ready() const noexcept {
  StateSnapshot snapshot;
  return snapshot_state(snapshot);
}

bool NativeUnitreeBackend::healthy(double state_absent_ms) const noexcept {
  StateSnapshot snapshot;
  if (!snapshot_state(snapshot)) {
    return false;
  }
  const double age_ms = static_cast<double>(
                            monotonic_ns_unitree() - snapshot.receive_ns) /
                        1.0e6;
  return age_ms <= state_absent_ms && !snapshot.faulted;
}

void NativeUnitreeBackend::begin_initialization(double duration_seconds) {
  if (!writes_enabled_) {
    throw std::runtime_error(
        "Unitree initialization needs explicitly enabled DDS writes");
  }
  if (require_realtime_ && !realtime_configured_.load()) {
    throw std::runtime_error(
        "Unitree writer did not obtain the required real-time settings");
  }
  if (mode_.load() != UnitreeMode::kDisabled ||
      hardware_fault_latched_.load(std::memory_order_acquire)) {
    throw std::runtime_error(
        "Unitree initialization requires a clean DISABLED state");
  }
  if (!healthy(state_absent_ms_) || duration_seconds <= 0.0) {
    throw std::runtime_error(
        "Unitree initialization needs fresh fault-free state and positive duration");
  }
  impl_->motion_switcher =
      std::make_unique<unitree::robot::b2::MotionSwitcherClient>();
  impl_->motion_switcher->SetTimeout(5.0F);
  impl_->motion_switcher->Init();
  std::string form;
  std::string name;
  impl_->motion_switcher->CheckMode(form, name);
  if (!name.empty()) {
    const int result = impl_->motion_switcher->ReleaseMode();
    if (result != 0) {
      throw std::runtime_error(
          "failed to release the active Unitree motion service: " + name);
    }
    for (int attempt = 0; attempt < 20; ++attempt) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      form.clear();
      name.clear();
      impl_->motion_switcher->CheckMode(form, name);
      if (name.empty()) {
        break;
      }
    }
    if (!name.empty()) {
      throw std::runtime_error(
          "Unitree motion service stayed active after release: " + name);
    }
  }
  StateSnapshot snapshot;
  if (!snapshot_state(snapshot) || snapshot.faulted) {
    throw std::runtime_error(
        "Unitree state faulted during motion-service release");
  }
  init_start_position_ = snapshot.joint_position;
  init_duration_seconds_ = duration_seconds;
  init_start_ns_ = monotonic_ns_unitree();
  mode_.store(UnitreeMode::kInitialize);
  if (hardware_fault_latched_.load(std::memory_order_acquire)) {
    mode_.store(UnitreeMode::kDamp);
    throw std::runtime_error(
        "Unitree state faulted before the initialization write gate opened");
  }
  write_gate_open_.store(true, std::memory_order_release);
}

void NativeUnitreeBackend::arm_control() {
  CommandSnapshot command;
  const bool command_ready = snapshot_command(command) &&
                             command.receive_ns != 0 &&
                             static_cast<double>(monotonic_ns_unitree() -
                                                 command.receive_ns) /
                                     1.0e6 <=
                                 command_stale_ms_;
  if (mode_.load() != UnitreeMode::kWait || !healthy(state_absent_ms_) ||
      !command_ready) {
    throw std::runtime_error(
        "Unitree control can arm only from WAIT with fresh state and command");
  }
  mode_.store(UnitreeMode::kControl);
}

void NativeUnitreeBackend::writer_loop() noexcept {
  bool configured = true;
  if (lock_memory_ && mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    configured = false;
  }
  if (writer_cpu_ >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(writer_cpu_, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
      configured = false;
    }
  }
  if (writer_fifo_priority_ > 0) {
    sched_param parameters{};
    parameters.sched_priority = writer_fifo_priority_;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &parameters) != 0) {
      configured = false;
    }
  }
  realtime_configured_.store(configured);
  if (!configured && require_realtime_) {
    mode_.store(UnitreeMode::kDisabled);
  }
  std::uint64_t scheduled_ns = monotonic_ns_unitree();
  LowCmd output;
  while (!stop_requested_.load(std::memory_order_relaxed)) {
    scheduled_ns += kWriterPeriodNs;
    const timespec target_time = unitree_timespec(scheduled_ns);
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &target_time,
                           nullptr) == EINTR) {
    }
    const std::uint64_t woke_ns = monotonic_ns_unitree();
    if (woke_ns > scheduled_ns) {
      update_unitree_max(wake_late_ns_max_, woke_ns - scheduled_ns);
    }
    writer_ticks_.fetch_add(1, std::memory_order_relaxed);
    const auto record_deadline = [this, scheduled_ns]() noexcept {
      if (monotonic_ns_unitree() > scheduled_ns + kWriterPeriodNs) {
        deadline_misses_.fetch_add(1, std::memory_order_relaxed);
      }
    };

    StateSnapshot state;
    if (!snapshot_state(state)) {
      record_deadline();
      continue;
    }
    CommandSnapshot command;
    snapshot_command(command);
    UnitreeMode mode = mode_.load();
    const double state_age_ms =
        static_cast<double>(woke_ns - state.receive_ns) / 1.0e6;
    if (mode != UnitreeMode::kDisabled && mode != UnitreeMode::kDamp &&
        state.faulted) {
      mode_.store(UnitreeMode::kDamp);
      mode = UnitreeMode::kDamp;
    } else if (mode != UnitreeMode::kDisabled &&
               mode != UnitreeMode::kDamp &&
               state_age_ms > state_absent_ms_) {
      watchdog_faults_.fetch_add(1, std::memory_order_relaxed);
      mode_.store(UnitreeMode::kDamp);
      mode = UnitreeMode::kDamp;
    }
    if (mode == UnitreeMode::kControl) {
      const double command_age_ms =
          command.receive_ns == 0
              ? command_stale_ms_ + 1.0
              : static_cast<double>(woke_ns - command.receive_ns) / 1.0e6;
      if (command_age_ms > command_stale_ms_) {
        watchdog_faults_.fetch_add(1, std::memory_order_relaxed);
        mode_.store(UnitreeMode::kDamp);
        mode = UnitreeMode::kDamp;
      }
    }
    if (mode == UnitreeMode::kInitialize) {
      const double ratio = std::clamp(
          static_cast<double>(woke_ns - init_start_ns_) /
              (init_duration_seconds_ * 1.0e9),
          0.0, 1.0);
      for (std::size_t index = 0; index < kJointCount; ++index) {
        command.joint_target[index] = static_cast<float>(
            init_start_position_[index] * (1.0 - ratio) +
            default_joint_position_[index] * ratio);
      }
      if (ratio >= 1.0) {
        mode_.store(UnitreeMode::kWait);
        mode = UnitreeMode::kWait;
      }
    } else if (mode == UnitreeMode::kWait) {
      command.joint_target = default_joint_position_;
    }
    if (mode == UnitreeMode::kDisabled ||
        !write_gate_open_.load(std::memory_order_acquire)) {
      record_deadline();
      continue;
    }

    output.mode_pr() = 0;
    output.mode_machine() = state.mode_machine;
    for (std::size_t isaac = 0; isaac < kJointCount; ++isaac) {
      const std::size_t sdk = isaac_to_sdk_[isaac];
      auto& motor = output.motor_cmd().at(sdk);
      motor.mode() = 1;
      motor.tau() = 0.0F;
      motor.dq() = 0.0F;
      if (mode == UnitreeMode::kDamp) {
        motor.q() = 0.0F;
        motor.kp() = 0.0F;
        motor.kd() = 8.0F;
      } else {
        motor.q() = command.joint_target[isaac];
        motor.kp() = stiffness_[isaac];
        motor.kd() = damping_[isaac];
      }
    }
    output.crc() = crc32_core(reinterpret_cast<std::uint32_t*>(&output),
                              (sizeof(output) >> 2) - 1);
    if (writes_enabled_) {
      bool published = false;
      try {
        published = impl_->publisher->Write(output);
      } catch (...) {
      }
      if (published) {
        publishes_.fetch_add(1, std::memory_order_relaxed);
      } else {
        publish_failures_.fetch_add(1, std::memory_order_relaxed);
        watchdog_faults_.fetch_add(1, std::memory_order_relaxed);
        mode_.store(UnitreeMode::kDamp);
      }
    }
    record_deadline();
  }
}

UnitreeWriterStats NativeUnitreeBackend::writer_stats() const noexcept {
  return UnitreeWriterStats{
      .ticks = writer_ticks_.load(),
      .publishes = publishes_.load(),
      .publish_failures = publish_failures_.load(),
      .crc_errors = crc_errors_.load(),
      .hardware_faults = hardware_faults_.load(),
      .watchdog_faults = watchdog_faults_.load(),
      .wake_late_ns_max = wake_late_ns_max_.load(),
      .deadline_misses = deadline_misses_.load(),
      .mode = mode_.load(),
      .writes_enabled = writes_enabled_,
      .realtime_configured = realtime_configured_.load(),
  };
}

}  // namespace ec_native
