#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <thread>

#include "native_backend.hpp"

namespace ec_native {

enum class UnitreeMode : std::uint32_t {
  kDisabled = 0,
  kInitialize = 1,
  kWait = 2,
  kControl = 3,
  kDamp = 4,
  // Operator-owned freeze on the last commanded target: the state the hoist
  // is hooked in. Only the state watchdogs leave it; a stale command does
  // not, because the control thread is stopped while the robot is held.
  kHold = 5,
};

struct UnitreeWriterStats {
  std::uint64_t ticks = 0;
  std::uint64_t publishes = 0;
  std::uint64_t publish_failures = 0;
  std::uint64_t crc_errors = 0;
  std::uint64_t hardware_faults = 0;
  // First tripped state guard, latched: 1 non-finite, 2 joint limit,
  // 4 joint speed, 8 motor state, 16 temperature, 32 IMU quaternion norm,
  // 64 pelvis tilt past 60 deg (a fall).
  std::uint32_t state_fault_reason = 0;
  std::uint32_t state_fault_joint = 0;
  std::uint32_t state_fault_sdk_joint = 0;
  std::uint32_t state_fault_motorstate = 0;
  std::uint64_t watchdog_faults = 0;
  std::uint64_t wake_late_ns_max = 0;
  std::uint64_t deadline_misses = 0;
  std::uint64_t ramp_faults = 0;
  // rt/lowstate arrival: frames seen and the longest gap between two, so a
  // state-absent watchdog trip says whether the wire actually went quiet.
  std::uint64_t state_frames = 0;
  std::uint64_t state_gap_ns_max = 0;
  float ramp_error_max = 0.0F;
  std::uint32_t ramp_error_joint = 0;
  float joint_speed_max = 0.0F;
  float tracking_error_max = 0.0F;
  // In WAIT: how far the control thread's (unapplied) target sits from the
  // held pose - the policy's first action, measured before it is applied.
  float command_target_error_max = 0.0F;
  std::uint32_t blend_ticks_remaining = 0;
  // The yaw the fixed initial anchor captured at this runtime start: the
  // reference start frame's heading minus the robot's, in degrees, wrapped
  // to (-180, 180]. The robot may boot facing any direction; this is by how
  // much, and it stays constant for the episode.
  float anchor_yaw_offset_degrees = 0.0F;
  bool anchor_heading_captured = false;
  UnitreeMode mode = UnitreeMode::kDisabled;
  bool writes_enabled = false;
  bool realtime_configured = false;
  bool gate_open = false;
  bool vendor_released = false;
  bool hardware_fault_latched = false;
};

// Native Unitree G1 DDS backend. DDS receive and the 500 Hz command writer
// stay outside Python. Writes are disabled by default and need an explicit
// initialization plus engage sequence.
class NativeUnitreeBackend final : public NativeRobotBackend {
 public:
  NativeUnitreeBackend(
      const std::string& network_interface,
      std::span<const std::size_t> isaac_to_sdk,
      std::span<const float> default_joint_position,
      std::span<const float> stiffness, std::span<const float> damping,
      std::span<const float> joint_lower, std::span<const float> joint_upper,
      bool writes_enabled, int writer_cpu, int writer_fifo_priority,
      bool lock_memory, bool require_realtime, double state_absent_ms,
      double command_stale_ms, int dds_domain = 0);
  ~NativeUnitreeBackend() override;

  NativeUnitreeBackend(const NativeUnitreeBackend&) = delete;
  NativeUnitreeBackend& operator=(const NativeUnitreeBackend&) = delete;

  void reset() override;
  const RobotState& read_state() noexcept override;
  void write_target(std::span<const float> target) noexcept override;
  void damp() noexcept override;
  void wait() noexcept override {}
  bool healthy(double state_absent_ms) const noexcept override;
  float blend_weight() const noexcept override;
  bool held_target(std::span<float> out) const noexcept override;

  bool state_ready() const noexcept;
  // Anchor the encoder at the reference's start-frame pose. The G1 has no
  // external localization, so position feedback stays at the selected
  // reference start frame even for a moving reference. The orientation
  // is expressed in the reference world using a constant yaw-only offset
  // captured at each runtime start. Absolute tilt and subsequent heading
  // changes remain visible. Using raw IMU yaw made the policy chase the
  // dataset's heading; matching full initial orientation hid real tilt error.
  void set_fixed_anchor_pose(std::span<const float> position,
                             std::span<const float> quaternion_xyzw);
  // hold_current ramps to the pose the robot is already in instead of the
  // bundle's default stance. The default stance is the hardware sequence; a
  // rehearsal episode that must start ON a reference frame holds instead, so
  // the init ramp does not drag the robot off that frame.
  // skip_motion_switcher drops the Unitree motion-service release handshake.
  // Against the simulated DDS plant there is no motion service, and the
  // client's CheckMode blocks for its whole timeout - long enough for a
  // planner-driven controller to starve and damp before it can arm. Never
  // skip it against real hardware: it is what stops the factory controller
  // and ours from writing at the same time.
  // A non-empty target_position (Isaac joint order) ramps to that pose
  // instead of the bundle default stance or the current pose - this is how the
  // operator sends the robot to a predefined qpos. It is validated against the
  // joint limits and a ramp-speed cap, because a large step over a short
  // duration is a thrown robot, not a slow one.
  void begin_initialization(double duration_seconds,
                            std::span<const float> target_position = {},
                            bool hold_current = false,
                            bool skip_motion_switcher = false);
  // The lifecycle sequence, split out of begin_initialization so takeover
  // is gapless: our damp frames are on the wire (open_damp_gate) before the
  // vendor lets go (release_vendor), and the vendor comes back only after
  // the wire is silent again (close_gate, restore_vendor).
  std::string vendor_mode();
  void open_damp_gate();
  void release_vendor();
  void hold();
  void close_gate();
  void restore_vendor(const std::string& name);
  // blend_ticks > 0 ramps the applied target from the held pose to the
  // policy's over that many writer ticks, so the first policy action is not
  // a step function against the hold.
  void engage_control(std::size_t blend_ticks = 0);
  // The ramp guard: a joint lagging the ramp by more than `rad` for more
  // than `ticks` writer ticks (2 ms each) is blocked, not slow. Tune per
  // bundle: weak-gain joints sag under gravity and need a looser bound.
  void set_ramp_guard(float rad, std::uint32_t ticks);
  // Gains for INITIALIZE / WAIT / HOLD: the policy's stiffness times `scale`
  // (damping times sqrt(scale), same damping ratio). The policy's own PD
  // gains let a held pose sag under gravity (0.4 rad on a 28 N m/rad waist),
  // so the start pose would not match the sim frame the policy expects; a
  // stiffer hold does, and the blend-in ramps the gains down with the target.
  void set_hold_gain_scale(float scale);
  // Drop a latched hardware fault so a new episode can PRECHECK after a fall.
  // The next rt/lowstate frame re-latches immediately if the robot is still
  // outside its limits.
  void clear_latched_fault() noexcept;
  void force_damp() noexcept;
  // A consistent seqlock snapshot of the latest rt/lowstate for the operator
  // thread. read_state() is the control thread's and updates its cache only
  // while that thread runs, so the lifecycle's pose gates must not use it.
  struct LatestState {
    std::array<float, kJointCount> joint_position{};
    std::array<float, kJointCount> joint_velocity{};
    std::array<float, 3> projected_gravity{0.0F, 0.0F, -1.0F};
    bool valid = false;
  };
  LatestState latest_state() const noexcept;
  // Per-joint |control-thread target - held pose| from the last WAIT tick.
  std::array<float, kJointCount> command_target_error() const noexcept;
  UnitreeMode mode() const noexcept { return mode_.load(); }
  UnitreeWriterStats writer_stats() const noexcept;

 private:
  struct Impl;
  struct StateSlot;
  struct StateSnapshot;
  struct CommandSlot;
  struct CommandSnapshot;

  void low_state_handler(const void* message) noexcept;
  void writer_loop() noexcept;
  void ensure_motion_switcher();
  void release_motion_service();
  bool snapshot_state(StateSnapshot& destination) const noexcept;
  bool snapshot_command(CommandSnapshot& destination) const noexcept;

  std::unique_ptr<Impl> impl_;
  std::array<std::size_t, kJointCount> isaac_to_sdk_{};
  std::array<float, kJointCount> default_joint_position_{};
  std::array<float, kJointCount> stiffness_{};
  std::array<float, kJointCount> damping_{};
  std::array<float, kJointCount> hold_stiffness_{};
  std::array<float, kJointCount> hold_damping_{};
  std::array<float, kJointCount> joint_lower_{};
  std::array<float, kJointCount> joint_upper_{};
  std::unique_ptr<StateSlot> state_slot_;
  std::unique_ptr<CommandSlot> command_slot_;
  RobotState state_cache_{};
  std::array<float, 3> fixed_anchor_position_{};
  // XYZW, matching RobotState::anchor_quaternion_w everywhere else.
  std::array<float, 4> fixed_anchor_quaternion_{0.0F, 0.0F, 0.0F, 1.0F};
  // IMU orientation (XYZW) at the first valid frame of each runtime start.
  std::array<float, 4> fixed_anchor_imu_start_{0.0F, 0.0F, 0.0F, 1.0F};
  bool fixed_anchor_enabled_ = false;
  bool fixed_anchor_imu_captured_ = false;
  // Written by the control thread on capture, read by the operator thread.
  std::atomic<float> anchor_yaw_offset_degrees_{0.0F};
  std::atomic<bool> anchor_heading_captured_{false};
  std::array<float, kJointCount> init_start_position_{};
  std::array<float, kJointCount> init_target_position_{};
  std::array<float, kJointCount> hold_target_{};
  std::atomic<std::size_t> blend_ticks_{0};
  std::atomic<std::size_t> blend_progress_{0};
  std::uint32_t ramp_error_ticks_ = 0;
  std::atomic<float> ramp_fault_rad_{0.5F};
  std::atomic<std::uint32_t> ramp_fault_ticks_{50};
  std::atomic<float> ramp_error_max_{0.0F};
  std::atomic<std::uint32_t> ramp_error_joint_{0};
  std::atomic<bool> vendor_released_{false};
  std::atomic<UnitreeMode> mode_{UnitreeMode::kDisabled};
  std::atomic<bool> write_gate_open_{false};
  std::atomic<bool> stop_requested_{false};
  bool writes_enabled_ = false;
  int writer_cpu_ = -1;
  int writer_fifo_priority_ = 0;
  bool lock_memory_ = false;
  bool require_realtime_ = false;
  double state_absent_ms_ = 500.0;
  double command_stale_ms_ = 500.0;
  std::atomic<bool> realtime_configured_{false};
  std::atomic<bool> hardware_fault_latched_{false};
  double init_duration_seconds_ = 3.0;
  std::uint64_t init_start_ns_ = 0;
  std::thread writer_thread_;

  std::atomic<std::uint64_t> writer_ticks_{0};
  std::atomic<std::uint64_t> publishes_{0};
  std::atomic<std::uint64_t> publish_failures_{0};
  std::atomic<std::uint64_t> crc_errors_{0};
  std::atomic<std::uint64_t> hardware_faults_{0};
  // First tripped guard, latched: 1 non-finite, 2 joint limit, 4 joint speed,
  // 8 motor state, 16 temperature, 32 IMU quaternion norm.
  std::atomic<std::uint32_t> state_fault_reason_{0};
  std::atomic<std::uint32_t> state_fault_joint_{0};
  std::atomic<std::uint32_t> state_fault_sdk_joint_{0};
  std::atomic<std::uint32_t> state_fault_motorstate_{0};
  std::atomic<std::uint64_t> watchdog_faults_{0};
  std::atomic<std::uint64_t> wake_late_ns_max_{0};
  std::atomic<std::uint64_t> deadline_misses_{0};
  std::atomic<std::uint64_t> ramp_faults_{0};
  std::atomic<std::uint64_t> state_frames_{0};
  std::atomic<std::uint64_t> state_gap_ns_max_{0};
  std::atomic<float> joint_speed_max_{0.0F};
  std::atomic<float> tracking_error_max_{0.0F};
  std::atomic<float> command_target_error_max_{0.0F};
  std::array<std::atomic<float>, kJointCount> command_target_error_{};
};

}  // namespace ec_native
