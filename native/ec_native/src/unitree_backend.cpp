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

#include "dds_channel.hpp"

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
// The SDK's own init examples ramp at 0.5 rad/s; anything materially faster is
// a throw rather than a move.
constexpr double kInitRampSpeedLimit = 0.5;
constexpr int kLocalSnapshotAttempts = 8;
constexpr float kMotionSwitcherTimeoutSeconds = 5.0F;
// A pelvis tilted past 60 deg is a fall in progress, whatever the joints say;
// the guard reads projected gravity's z, which is -cos(tilt).
constexpr float kTiltFaultGravityZ = -0.5F;

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

// Ages are taken against a timestamp another thread may have stored a few
// hundred nanoseconds *after* `now` was sampled. Unsigned subtraction then
// wraps to 1.8e10 ms and trips a watchdog; that fired about once every 20 s
// of loopback and would have damped a walking robot for no reason.
double age_ms_since(std::uint64_t now_ns, std::uint64_t stamp_ns) noexcept {
  const std::int64_t delta = static_cast<std::int64_t>(now_ns) -
                             static_cast<std::int64_t>(stamp_ns);
  return delta > 0 ? static_cast<double>(delta) / 1.0e6 : 0.0;
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
    double command_stale_ms, int dds_domain)
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
  hold_stiffness_ = stiffness_;
  hold_damping_ = damping_;
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

  // Domain 0 is the robot; a simulated plant pair may use another domain.
  ensure_channel_factory(dds_domain, network_interface);
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

void NativeUnitreeBackend::reset() {
  // Probe, go, and subsequent episodes can start at different headings.
  // NativeFakeRuntime calls reset only after the previous loop has joined.
  fixed_anchor_imu_captured_ = false;
  state_cache_.anchor_pose_valid = false;
}

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
  // The joint limits are the policy's training limits, not the actuators'.
  // A limp robot hanging under vendor damp folds its torso past the waist's
  // 0.52 rad soft limit; that is not a fault until we hold or drive a target
  // inside those limits. The init ramp starts from wherever the joints are
  // and its own tracking guard covers it.
  const UnitreeMode guard_mode = mode_.load();
  const bool enforce_soft_limits = guard_mode == UnitreeMode::kWait ||
                                   guard_mode == UnitreeMode::kControl ||
                                   guard_mode == UnitreeMode::kHold;
  for (std::size_t isaac = 0; isaac < kJointCount; ++isaac) {
    const std::size_t sdk = isaac_to_sdk_[isaac];
    const auto& motor = input.motor_state()[sdk];
    state_slot_->joint_position[isaac].store(motor.q(),
                                              std::memory_order_relaxed);
    state_slot_->joint_velocity[isaac].store(motor.dq(),
                                              std::memory_order_relaxed);
    // Reason bits so a latched fault says WHICH guard tripped; a DAMP with no
    // cause is unactionable on hardware and unreadable in a rehearsal log.
    std::uint32_t reason = 0;
    if (!std::isfinite(motor.q()) || !std::isfinite(motor.dq())) {
      reason |= 1U;
    }
    if (enforce_soft_limits &&
        (motor.q() < joint_lower_[isaac] - kJointLimitFaultMargin ||
         motor.q() > joint_upper_[isaac] + kJointLimitFaultMargin)) {
      reason |= 2U;
    }
    if (std::abs(motor.dq()) > 35.0F) {
      reason |= 4U;
    }
    if (motor.motorstate() != 0) {
      reason |= 8U;
    }
    if (std::max(motor.temperature()[0], motor.temperature()[1]) >= 90) {
      reason |= 16U;
    }
    if (reason != 0U) {
      std::uint32_t expected = 0U;
      state_fault_reason_.compare_exchange_strong(expected, reason);
      std::uint32_t expected_joint = 0U;
      state_fault_joint_.compare_exchange_strong(
          expected_joint, static_cast<std::uint32_t>(isaac));
    }
    faulted = faulted || reason != 0U;
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
  if (quaternion_norm_squared < 0.5F || quaternion_norm_squared > 1.5F) {
    std::uint32_t expected = 0U;
    state_fault_reason_.compare_exchange_strong(expected, 32U);
    faulted = true;
  } else {
    const float w = input_quaternion[0];
    const float x = input_quaternion[1];
    const float y = input_quaternion[2];
    const float gravity_z = -(1.0F - 2.0F * (x * x + y * y));
    static_cast<void>(w);
    if (gravity_z > kTiltFaultGravityZ) {
      std::uint32_t expected = 0U;
      state_fault_reason_.compare_exchange_strong(expected, 64U);
      faulted = true;
    }
  }
  const std::uint64_t now_ns = monotonic_ns_unitree();
  const std::uint64_t previous_ns =
      state_slot_->receive_ns.load(std::memory_order_relaxed);
  if (previous_ns != 0 && now_ns > previous_ns) {
    update_unitree_max(state_gap_ns_max_, now_ns - previous_ns);
  }
  state_frames_.fetch_add(1, std::memory_order_relaxed);
  state_slot_->receive_ns.store(now_ns, std::memory_order_relaxed);
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
    // A DISABLED writer publishes nothing; re-damping it would make
    // close_gate impossible while the fallen robot keeps reporting limits.
    if (mode_.load() != UnitreeMode::kDisabled) {
      mode_.store(UnitreeMode::kDamp);
    }
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
  // One projected-gravity implementation for every backend. A hand-written
  // copy here had roll and pitch sign-flipped: the plant rehearsal showed the
  // policy leaning into every tilt until it fell, four seconds after engaging.
  const std::array<float, 4> quaternion_xyzw = {
      snapshot.quaternion[1], snapshot.quaternion[2], snapshot.quaternion[3],
      snapshot.quaternion[0]};
  if (!projected_gravity_from_xyzw(quaternion_xyzw,
                                   state_cache_.projected_gravity)) {
    state_cache_.projected_gravity = {0.0F, 0.0F, -1.0F};
  }
  if (fixed_anchor_enabled_) {
    if (!fixed_anchor_imu_captured_) {
      fixed_anchor_imu_start_ = quaternion_xyzw;
      fixed_anchor_imu_captured_ = true;
    }
    state_cache_.anchor_position_w = fixed_anchor_position_;
    state_cache_.anchor_pose_valid = align_heading_to_reference(
        fixed_anchor_imu_start_, fixed_anchor_quaternion_, quaternion_xyzw,
        state_cache_.anchor_quaternion_w);
  }
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
  // HOLD is operator-owned: the control thread is stopped while the robot is
  // held, so a stale-command fault from it must not drop a held robot.
  if (mode == UnitreeMode::kWait || mode == UnitreeMode::kControl) {
    mode_.store(UnitreeMode::kDamp);
  }
}

void NativeUnitreeBackend::force_damp() noexcept {
  mode_.store(UnitreeMode::kDamp);
}

float NativeUnitreeBackend::blend_weight() const noexcept {
  if (mode_.load() != UnitreeMode::kControl) {
    return 1.0F;
  }
  const std::size_t ticks = blend_ticks_.load(std::memory_order_relaxed);
  const std::size_t progress = blend_progress_.load(std::memory_order_relaxed);
  if (ticks == 0 || progress >= ticks) {
    return 1.0F;
  }
  return static_cast<float>(progress) / static_cast<float>(ticks);
}

bool NativeUnitreeBackend::held_target(std::span<float> out) const noexcept {
  if (out.size() != kJointCount) {
    return false;
  }
  std::copy(init_target_position_.begin(), init_target_position_.end(),
            out.begin());
  return true;
}

void NativeUnitreeBackend::set_hold_gain_scale(float scale) {
  if (!(scale > 0.0F) || !std::isfinite(scale)) {
    throw std::runtime_error("hold gain scale must be positive");
  }
  const UnitreeMode current = mode_.load();
  if (current != UnitreeMode::kDisabled && current != UnitreeMode::kDamp) {
    throw std::runtime_error(
        "hold gains can only change while DISABLED or in DAMP");
  }
  const float damping_scale = std::sqrt(scale);
  for (std::size_t index = 0; index < kJointCount; ++index) {
    hold_stiffness_[index] = stiffness_[index] * scale;
    hold_damping_[index] = damping_[index] * damping_scale;
  }
}

void NativeUnitreeBackend::clear_latched_fault() noexcept {
  state_slot_->faulted.store(false, std::memory_order_relaxed);
  state_fault_reason_.store(0);
  state_fault_joint_.store(0);
  hardware_fault_latched_.store(false, std::memory_order_release);
}

void NativeUnitreeBackend::set_ramp_guard(float rad, std::uint32_t ticks) {
  if (!(rad > 0.0F) || !std::isfinite(rad) || ticks == 0) {
    throw std::runtime_error(
        "ramp guard needs a positive tolerance and tick count");
  }
  ramp_fault_rad_.store(rad);
  ramp_fault_ticks_.store(ticks);
}

std::array<float, kJointCount> NativeUnitreeBackend::command_target_error()
    const noexcept {
  std::array<float, kJointCount> out{};
  for (std::size_t index = 0; index < kJointCount; ++index) {
    out[index] = command_target_error_[index].load(std::memory_order_relaxed);
  }
  return out;
}

NativeUnitreeBackend::LatestState NativeUnitreeBackend::latest_state() const
    noexcept {
  LatestState out;
  StateSnapshot snapshot;
  if (!snapshot_state(snapshot)) {
    return out;
  }
  out.joint_position = snapshot.joint_position;
  out.joint_velocity = snapshot.joint_velocity;
  const std::array<float, 4> quaternion_xyzw = {
      snapshot.quaternion[1], snapshot.quaternion[2], snapshot.quaternion[3],
      snapshot.quaternion[0]};
  if (!projected_gravity_from_xyzw(quaternion_xyzw, out.projected_gravity)) {
    out.projected_gravity = {0.0F, 0.0F, -1.0F};
  }
  out.valid = true;
  return out;
}

bool NativeUnitreeBackend::state_ready() const noexcept {
  StateSnapshot snapshot;
  return snapshot_state(snapshot);
}

void NativeUnitreeBackend::set_fixed_anchor_pose(
    std::span<const float> position, std::span<const float> quaternion_xyzw) {
  if (position.size() != fixed_anchor_position_.size() ||
      !finite_values(position) ||
      quaternion_xyzw.size() != fixed_anchor_quaternion_.size() ||
      !finite_values(quaternion_xyzw)) {
    throw std::runtime_error(
        "fixed Unitree anchor pose must be 3 + 4 finite values");
  }
  float norm_squared = 0.0F;
  for (const float value : quaternion_xyzw) {
    norm_squared += value * value;
  }
  if (norm_squared < 0.5F || norm_squared > 1.5F) {
    throw std::runtime_error("fixed Unitree anchor quaternion is not unit");
  }
  std::copy(position.begin(), position.end(), fixed_anchor_position_.begin());
  std::copy(quaternion_xyzw.begin(), quaternion_xyzw.end(),
            fixed_anchor_quaternion_.begin());
  fixed_anchor_enabled_ = true;
  fixed_anchor_imu_captured_ = false;
}

bool NativeUnitreeBackend::healthy(double state_absent_ms) const noexcept {
  // DDS can begin another 1 kHz update during every bounded snapshot retry.
  // That is contention, not a lost robot state stream.
  const std::uint64_t receive_ns =
      state_slot_->receive_ns.load(std::memory_order_acquire);
  if (receive_ns == 0) {
    return false;
  }
  const double age_ms = age_ms_since(monotonic_ns_unitree(), receive_ns);
  return age_ms <= state_absent_ms &&
         !hardware_fault_latched_.load(std::memory_order_acquire);
}

void NativeUnitreeBackend::ensure_motion_switcher() {
  if (impl_->motion_switcher) {
    return;
  }
  impl_->motion_switcher =
      std::make_unique<unitree::robot::b2::MotionSwitcherClient>();
  impl_->motion_switcher->SetTimeout(kMotionSwitcherTimeoutSeconds);
  impl_->motion_switcher->Init();
}

void NativeUnitreeBackend::release_motion_service() {
  ensure_motion_switcher();
  std::string form;
  std::string name;
  impl_->motion_switcher->CheckMode(form, name);
  if (!name.empty()) {
    const int result = impl_->motion_switcher->ReleaseMode();
    if (result != 0) {
      throw std::runtime_error(
          "failed to release the active Unitree motion service: " + name +
          " (status " + std::to_string(result) + ")");
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
  vendor_released_.store(true);
}

std::string NativeUnitreeBackend::vendor_mode() {
  ensure_motion_switcher();
  std::string form;
  std::string name;
  impl_->motion_switcher->CheckMode(form, name);
  return name;
}

void NativeUnitreeBackend::open_damp_gate() {
  if (!writes_enabled_) {
    throw std::runtime_error(
        "opening the Unitree write gate needs explicitly enabled DDS writes");
  }
  if (require_realtime_ && !realtime_configured_.load()) {
    throw std::runtime_error(
        "Unitree writer did not obtain the required real-time settings");
  }
  if (hardware_fault_latched_.load(std::memory_order_acquire)) {
    throw std::runtime_error(
        "Unitree write gate refused: a hardware fault is latched");
  }
  const UnitreeMode current = mode_.load();
  if (current != UnitreeMode::kDisabled && current != UnitreeMode::kDamp) {
    throw std::runtime_error(
        "open_damp_gate needs the writer DISABLED or in DAMP");
  }
  if (!healthy(state_absent_ms_)) {
    throw std::runtime_error(
        "open_damp_gate needs fresh fault-free robot state");
  }
  mode_.store(UnitreeMode::kDamp);
  write_gate_open_.store(true, std::memory_order_release);
}

void NativeUnitreeBackend::release_vendor() {
  if (!write_gate_open_.load(std::memory_order_acquire) ||
      mode_.load() != UnitreeMode::kDamp) {
    throw std::runtime_error(
        "release_vendor needs our damp frames on the wire first "
        "(open_damp_gate in DAMP)");
  }
  release_motion_service();
}

void NativeUnitreeBackend::hold() {
  const UnitreeMode current = mode_.load();
  if (current == UnitreeMode::kControl) {
    CommandSnapshot command;
    if (!snapshot_command(command)) {
      throw std::runtime_error("hold could not read the last command target");
    }
    hold_target_ = command.joint_target;
  } else if (current == UnitreeMode::kWait) {
    hold_target_ = init_target_position_;
  } else {
    throw std::runtime_error("hold needs the writer in CONTROL or WAIT");
  }
  mode_.store(UnitreeMode::kHold);
}

void NativeUnitreeBackend::close_gate() {
  const UnitreeMode current = mode_.load();
  if (current != UnitreeMode::kDamp && current != UnitreeMode::kDisabled) {
    throw std::runtime_error("close_gate needs the writer in DAMP");
  }
  write_gate_open_.store(false, std::memory_order_release);
  mode_.store(UnitreeMode::kDisabled);
  blend_ticks_.store(0);
  blend_progress_.store(0);
  ramp_error_ticks_ = 0;
}

void NativeUnitreeBackend::restore_vendor(const std::string& name) {
  if (name.empty()) {
    throw std::runtime_error("restore_vendor needs the motion service name");
  }
  if (write_gate_open_.load(std::memory_order_acquire) ||
      mode_.load() != UnitreeMode::kDisabled) {
    throw std::runtime_error(
        "restore_vendor needs the write gate closed (close_gate first): two "
        "controllers must never publish rt/lowcmd at once");
  }
  ensure_motion_switcher();
  const int result = impl_->motion_switcher->SelectMode(name);
  if (result != 0) {
    throw std::runtime_error("failed to select Unitree motion service '" +
                             name + "' (status " + std::to_string(result) +
                             ")");
  }
  for (int attempt = 0; attempt < 30; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::string form;
    std::string current;
    impl_->motion_switcher->CheckMode(form, current);
    if (current == name) {
      vendor_released_.store(false);
      return;
    }
  }
  throw std::runtime_error("Unitree motion service '" + name +
                           "' did not report active after SelectMode");
}

void NativeUnitreeBackend::begin_initialization(
    double duration_seconds, std::span<const float> target_position,
    bool hold_current, bool skip_motion_switcher) {
  if (!writes_enabled_) {
    throw std::runtime_error(
        "Unitree initialization needs explicitly enabled DDS writes");
  }
  if (require_realtime_ && !realtime_configured_.load()) {
    throw std::runtime_error(
        "Unitree writer did not obtain the required real-time settings");
  }
  if (hardware_fault_latched_.load(std::memory_order_acquire)) {
    throw std::runtime_error(
        "Unitree initialization refused: a hardware fault is latched");
  }
  const UnitreeMode current = mode_.load();
  const bool gate_open = write_gate_open_.load(std::memory_order_acquire);
  const bool from_disabled = current == UnitreeMode::kDisabled && !gate_open;
  const bool from_open_gate =
      gate_open && (current == UnitreeMode::kDamp ||
                    current == UnitreeMode::kHold ||
                    current == UnitreeMode::kWait);
  if (!from_disabled && !from_open_gate) {
    throw std::runtime_error(
        "Unitree initialization requires a clean DISABLED state, or an open "
        "write gate in DAMP, HOLD or WAIT");
  }
  if (!healthy(state_absent_ms_) || duration_seconds <= 0.0) {
    throw std::runtime_error(
        "Unitree initialization needs fresh fault-free state and positive duration");
  }
  if (from_disabled) {
    if (!skip_motion_switcher) {
      release_motion_service();
    }
  } else if (!skip_motion_switcher && !vendor_released_.load()) {
    throw std::runtime_error(
        "the vendor motion service still owns the joints; call "
        "release_vendor first");
  }
  StateSnapshot snapshot;
  if (!snapshot_state(snapshot) || snapshot.faulted) {
    throw std::runtime_error(
        "Unitree state faulted during motion-service release");
  }
  init_start_position_ = snapshot.joint_position;
  if (!target_position.empty()) {
    if (target_position.size() != kJointCount || !finite_values(target_position)) {
      throw std::runtime_error(
          "Unitree initialization target must be 29 finite values");
    }
    for (std::size_t isaac = 0; isaac < kJointCount; ++isaac) {
      const float target = target_position[isaac];
      if (target < joint_lower_[isaac] - kJointLimitFaultMargin ||
          target > joint_upper_[isaac] + kJointLimitFaultMargin) {
        throw std::runtime_error(
            "Unitree initialization target exceeds the joint limit at index " +
            std::to_string(isaac));
      }
      const double speed =
          std::abs(static_cast<double>(target) - snapshot.joint_position[isaac]) /
          duration_seconds;
      if (speed > kInitRampSpeedLimit) {
        throw std::runtime_error(
            "Unitree initialization ramp exceeds " +
            std::to_string(kInitRampSpeedLimit) + " rad/s at joint index " +
            std::to_string(isaac) + "; lengthen the duration");
      }
    }
    std::copy(target_position.begin(), target_position.end(),
              init_target_position_.begin());
  } else {
    init_target_position_ =
        hold_current ? snapshot.joint_position : default_joint_position_;
  }
  init_duration_seconds_ = duration_seconds;
  init_start_ns_ = monotonic_ns_unitree();
  ramp_error_ticks_ = 0;
  ramp_error_max_.store(0.0F);
  ramp_error_joint_.store(0);
  blend_ticks_.store(0);
  blend_progress_.store(0);
  mode_.store(UnitreeMode::kInitialize);
  if (hardware_fault_latched_.load(std::memory_order_acquire)) {
    mode_.store(UnitreeMode::kDamp);
    throw std::runtime_error(
        "Unitree state faulted before the initialization write gate opened");
  }
  write_gate_open_.store(true, std::memory_order_release);
}

void NativeUnitreeBackend::engage_control(std::size_t blend_ticks) {
  CommandSnapshot command;
  const bool command_ready =
      snapshot_command(command) && command.receive_ns != 0 &&
      age_ms_since(monotonic_ns_unitree(), command.receive_ns) <=
          command_stale_ms_;
  if (mode_.load() != UnitreeMode::kWait || !healthy(state_absent_ms_) ||
      !command_ready) {
    throw std::runtime_error(
        "Unitree control can arm only from WAIT with fresh state and command");
  }
  blend_ticks_.store(blend_ticks);
  blend_progress_.store(0);
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

    const std::uint64_t state_receive_ns =
        state_slot_->receive_ns.load(std::memory_order_acquire);
    if (state_receive_ns == 0) {
      record_deadline();
      continue;
    }
    const bool state_faulted =
        hardware_fault_latched_.load(std::memory_order_acquire);
    const std::uint8_t mode_machine =
        state_slot_->mode_machine.load(std::memory_order_acquire);
    CommandSnapshot command;
    snapshot_command(command);
    UnitreeMode mode = mode_.load();
    const double state_age_ms = age_ms_since(woke_ns, state_receive_ns);
    if (mode != UnitreeMode::kDisabled && mode != UnitreeMode::kDamp &&
        state_faulted) {
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
          command.receive_ns == 0 ? command_stale_ms_ + 1.0
                                  : age_ms_since(woke_ns, command.receive_ns);
      if (command_age_ms > command_stale_ms_) {
        watchdog_faults_.fetch_add(1, std::memory_order_relaxed);
        mode_.store(UnitreeMode::kDamp);
        mode = UnitreeMode::kDamp;
      }
    }
    // Relaxed per-joint loads, not a seqlock snapshot: a torn read across
    // joints cannot matter to guards that need many consecutive ticks, and
    // the 1 kHz state handler must never stall this deadline.
    float joint_speed_max = 0.0F;
    for (std::size_t index = 0; index < kJointCount; ++index) {
      joint_speed_max = std::max(
          joint_speed_max,
          std::abs(state_slot_->joint_velocity[index].load(
              std::memory_order_relaxed)));
    }
    joint_speed_max_.store(joint_speed_max, std::memory_order_relaxed);
    // 1 = hold gains, 0 = the policy's; the blend-in walks it down with the
    // target so gains and target hand over together.
    float hold_weight = 0.0F;
    if (mode == UnitreeMode::kInitialize) {
      hold_weight = 1.0F;
      const double ratio = std::clamp(
          static_cast<double>(woke_ns - init_start_ns_) /
              (init_duration_seconds_ * 1.0e9),
          0.0, 1.0);
      float ramp_error = 0.0F;
      std::uint32_t ramp_joint = 0;
      for (std::size_t index = 0; index < kJointCount; ++index) {
        command.joint_target[index] = static_cast<float>(
            init_start_position_[index] * (1.0 - ratio) +
            init_target_position_[index] * ratio);
        const float error =
            std::abs(state_slot_->joint_position[index].load(
                         std::memory_order_relaxed) -
                     command.joint_target[index]);
        if (error > ramp_error) {
          ramp_error = error;
          ramp_joint = static_cast<std::uint32_t>(index);
        }
      }
      if (ramp_error > ramp_error_max_.load(std::memory_order_relaxed)) {
        ramp_error_max_.store(ramp_error, std::memory_order_relaxed);
        ramp_error_joint_.store(ramp_joint, std::memory_order_relaxed);
      }
      if (ramp_error > ramp_fault_rad_.load(std::memory_order_relaxed)) {
        if (++ramp_error_ticks_ >
            ramp_fault_ticks_.load(std::memory_order_relaxed)) {
          ramp_faults_.fetch_add(1, std::memory_order_relaxed);
          watchdog_faults_.fetch_add(1, std::memory_order_relaxed);
          mode_.store(UnitreeMode::kDamp);
          mode = UnitreeMode::kDamp;
        }
      } else {
        ramp_error_ticks_ = 0;
      }
      if (mode == UnitreeMode::kInitialize && ratio >= 1.0) {
        mode_.store(UnitreeMode::kWait);
        mode = UnitreeMode::kWait;
      }
    } else if (mode == UnitreeMode::kWait) {
      hold_weight = 1.0F;
      float command_error = 0.0F;
      if (command.receive_ns != 0) {
        for (std::size_t index = 0; index < kJointCount; ++index) {
          const float error = std::abs(command.joint_target[index] -
                                       init_target_position_[index]);
          command_target_error_[index].store(error, std::memory_order_relaxed);
          command_error = std::max(command_error, error);
        }
      }
      command_target_error_max_.store(command_error,
                                      std::memory_order_relaxed);
      command.joint_target = init_target_position_;
    } else if (mode == UnitreeMode::kHold) {
      hold_weight = 1.0F;
      command.joint_target = hold_target_;
    } else if (mode == UnitreeMode::kControl) {
      const std::size_t blend_ticks =
          blend_ticks_.load(std::memory_order_relaxed);
      const std::size_t progress =
          blend_progress_.load(std::memory_order_relaxed);
      // Gains blend here; the target blend lives on the control thread so
      // the policy's last_action reports what was applied.
      if (progress < blend_ticks) {
        const float weight = static_cast<float>(progress + 1) /
                             static_cast<float>(blend_ticks);
        hold_weight = 1.0F - weight;
        blend_progress_.store(progress + 1, std::memory_order_relaxed);
      }
    }
    if (mode != UnitreeMode::kDamp && mode != UnitreeMode::kDisabled) {
      float tracking_error = 0.0F;
      for (std::size_t index = 0; index < kJointCount; ++index) {
        tracking_error = std::max(
            tracking_error,
            std::abs(state_slot_->joint_position[index].load(
                         std::memory_order_relaxed) -
                     command.joint_target[index]));
      }
      tracking_error_max_.store(tracking_error, std::memory_order_relaxed);
    }
    if (mode == UnitreeMode::kDisabled ||
        !write_gate_open_.load(std::memory_order_acquire)) {
      record_deadline();
      continue;
    }

    output.mode_pr() = 0;
    output.mode_machine() = mode_machine;
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
        motor.kp() = hold_stiffness_[isaac] * hold_weight +
                     stiffness_[isaac] * (1.0F - hold_weight);
        motor.kd() = hold_damping_[isaac] * hold_weight +
                     damping_[isaac] * (1.0F - hold_weight);
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
      .state_fault_reason = state_fault_reason_.load(),
      .state_fault_joint = state_fault_joint_.load(),
      .watchdog_faults = watchdog_faults_.load(),
      .wake_late_ns_max = wake_late_ns_max_.load(),
      .deadline_misses = deadline_misses_.load(),
      .ramp_faults = ramp_faults_.load(),
      .state_frames = state_frames_.load(),
      .state_gap_ns_max = state_gap_ns_max_.load(),
      .ramp_error_max = ramp_error_max_.load(),
      .ramp_error_joint = ramp_error_joint_.load(),
      .joint_speed_max = joint_speed_max_.load(),
      .tracking_error_max = tracking_error_max_.load(),
      .command_target_error_max = command_target_error_max_.load(),
      .blend_ticks_remaining = static_cast<std::uint32_t>(
          blend_ticks_.load() > blend_progress_.load()
              ? blend_ticks_.load() - blend_progress_.load()
              : 0),
      .mode = mode_.load(),
      .writes_enabled = writes_enabled_,
      .realtime_configured = realtime_configured_.load(),
      .gate_open = write_gate_open_.load(std::memory_order_acquire),
      .vendor_released = vendor_released_.load(),
  };
}

}  // namespace ec_native
