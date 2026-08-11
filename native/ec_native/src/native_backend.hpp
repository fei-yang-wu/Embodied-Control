#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "native_tracker_core.hpp"

namespace ec_native {

bool projected_gravity_from_xyzw(std::span<const float> quaternion,
                                 std::span<float> gravity) noexcept;

struct NativeBackendTimingStats {
  std::uint64_t steps = 0;
  std::uint64_t wake_late_ns_max = 0;
  std::uint64_t deadline_misses = 0;
  bool realtime_configured = false;
};

class NativeRobotBackend {
 public:
  virtual ~NativeRobotBackend() = default;
  virtual void reset() = 0;
  // Optional frame-0 start: [root pos 3 | root quat XYZW 4 | joints 29].
  virtual void set_initial_pose(std::span<const float> pose) {
    static_cast<void>(pose);
  }
  virtual void start(bool paced) {
    static_cast<void>(paced);
  }
  virtual void stop() noexcept {}
  virtual void wait_for_stop() noexcept {}
  virtual bool requires_pacing() const noexcept { return false; }
  virtual const RobotState& read_state() noexcept = 0;
  virtual void write_target(std::span<const float> target) noexcept = 0;
  virtual void damp() noexcept = 0;
  virtual void wait() noexcept { damp(); }
  virtual double time() const noexcept { return 0.0; }
  virtual double base_height() const noexcept { return 0.0; }
  virtual double min_base_height() const noexcept { return base_height(); }
  virtual bool healthy(double state_absent_ms) const noexcept {
    static_cast<void>(state_absent_ms);
    return true;
  }
  virtual NativeBackendTimingStats timing_stats() const noexcept { return {}; }
};

class NativeFakeBackend final : public NativeRobotBackend {
 public:
  NativeFakeBackend(std::span<const float> default_joint_position,
                    std::size_t control_hz, float lag_alpha);
  void reset() override;
  const RobotState& read_state() noexcept override { return state_; }
  void write_target(std::span<const float> target) noexcept override;
  void damp() noexcept override;

 private:
  std::array<float, kJointCount> default_joint_position_{};
  std::size_t control_hz_ = 50;
  float lag_alpha_ = 1.0F;
  RobotState state_{};
};

class NativeMujocoBackend final : public NativeRobotBackend {
 public:
  NativeMujocoBackend(
      const std::string& model_path,
      const std::vector<std::string>& isaac_joint_names,
      std::span<const float> default_joint_position,
      std::span<const float> stiffness, std::span<const float> damping,
      std::span<const float> armature, std::span<const float> effort_limit,
      double timestep, std::size_t decimation, int physics_cpu = -1,
      int physics_fifo_priority = 0, bool lock_memory = false,
      bool require_realtime = false);
  ~NativeMujocoBackend() override;

  NativeMujocoBackend(const NativeMujocoBackend&) = delete;
  NativeMujocoBackend& operator=(const NativeMujocoBackend&) = delete;

  void reset() override;
  void set_initial_pose(std::span<const float> pose) override;
  void start(bool paced) override;
  void stop() noexcept override;
  void wait_for_stop() noexcept override;
  bool requires_pacing() const noexcept override { return true; }
  const RobotState& read_state() noexcept override;
  void write_target(std::span<const float> target) noexcept override;
  void damp() noexcept override;

  double simulation_time() const noexcept;
  double time() const noexcept override { return simulation_time(); }
  double base_height() const noexcept override;
  double min_base_height() const noexcept override;
  bool healthy(double state_absent_ms) const noexcept override;
  NativeBackendTimingStats timing_stats() const noexcept override;

 private:
  struct StateSlot;
  struct StateSnapshot;
  struct CommandSlot;
  struct CommandSnapshot;

  void set_gains(std::span<const float> stiffness,
                 std::span<const float> damping) noexcept;
  void physics_loop() noexcept;
  bool configure_physics_thread() noexcept;
  void publish_state() noexcept;
  bool snapshot_state(StateSnapshot& destination) const noexcept;
  bool snapshot_command(CommandSnapshot& destination) const noexcept;

  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::unique_ptr<StateSlot> state_slot_;
  std::unique_ptr<CommandSlot> command_slot_;
  std::array<float, kJointCount> default_joint_position_{};
  std::array<float, 36> initial_pose_{};
  bool has_initial_pose_ = false;
  std::array<float, kJointCount> stiffness_{};
  std::array<float, kJointCount> damping_{};
  std::array<std::size_t, kJointCount> actuator_to_isaac_{};
  std::array<int, kJointCount> qpos_address_{};
  std::array<int, kJointCount> dof_address_{};
  int pelvis_body_id_ = -1;
  double timestep_ = 0.005;
  std::size_t decimation_ = 4;
  int physics_cpu_ = -1;
  int physics_fifo_priority_ = 0;
  bool lock_memory_ = false;
  bool require_realtime_ = false;
  RobotState state_cache_{};
  std::thread physics_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> thread_ready_{false};
  std::atomic<bool> damp_requested_{false};
  std::atomic<bool> realtime_configured_{false};
  std::atomic<bool> physics_fault_{false};
  std::atomic<std::uint64_t> physics_steps_{0};
  std::atomic<std::uint64_t> wake_late_ns_max_{0};
  std::atomic<std::uint64_t> deadline_misses_{0};
  std::atomic<std::uint64_t> last_state_read_ns_{0};
  std::atomic<double> simulation_time_{0.0};
  std::atomic<float> base_height_{0.0F};
  std::atomic<float> min_base_height_{0.0F};
};

}  // namespace ec_native
