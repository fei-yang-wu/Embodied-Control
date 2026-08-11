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

struct PlantStats {
  std::uint64_t steps = 0;
  std::uint64_t publishes = 0;
  std::uint64_t publish_failures = 0;
  std::uint64_t commands_received = 0;
  std::uint64_t crc_errors = 0;
  std::uint64_t wake_late_ns_max = 0;
  std::uint64_t deadline_misses = 0;
  bool holding = true;
  bool physics_fault = false;
  bool realtime_configured = false;
  double last_command_age_ms = -1.0;
  double time = 0.0;
  double base_height = 0.0;
  double min_base_height = 0.0;
  double applied_kp_min = -1.0;
  double applied_kp_max = -1.0;
  double applied_kd_min = -1.0;
  double applied_kd_max = -1.0;
  double applied_q_absmax = -1.0;
  double applied_extra_absmax = -1.0;
};

// The sim side of the Digit-style unified plant interface: a MuJoCo process
// that serves the exact Unitree G1 DDS wire protocol the real robot serves
// (rt/lowstate out, rt/lowcmd in, hg IDL, CRC-checked), so the controller
// runs one hardware code path and only the network interface changes
// ("lo" against this plant, the robot NIC against hardware). The plant is
// SDK-native: every per-joint array is in SDK motor order and it knows
// nothing about the Isaac ordering used elsewhere in the runtime.
class MujocoDdsPlant {
 public:
  MujocoDdsPlant(const std::string& model_path,
                 const std::string& network_interface,
                 const std::vector<std::string>& sdk_joint_names,
                 std::span<const float> default_joint_position,
                 std::span<const float> armature,
                 std::span<const float> effort_limit,
                 std::span<const float> hold_stiffness,
                 std::span<const float> hold_damping, double timestep,
                 std::uint8_t mode_machine, int physics_cpu = -1,
                 int physics_fifo_priority = 0, bool lock_memory = false,
                 bool require_realtime = false);
  ~MujocoDdsPlant();

  MujocoDdsPlant(const MujocoDdsPlant&) = delete;
  MujocoDdsPlant& operator=(const MujocoDdsPlant&) = delete;

  void reset();
  // Optional start pose: [root pos 3 | root quat XYZW 4 | joints 29 SDK order].
  void set_initial_pose(std::span<const float> pose);
  void start();
  void stop() noexcept;
  void wait_for_stop() noexcept;
  bool running() const noexcept { return running_.load(); }
  PlantStats stats() const noexcept;

 private:
  struct Impl;
  struct CommandSlot;
  struct CommandSnapshot;

  void low_cmd_handler(const void* message) noexcept;
  bool snapshot_command(CommandSnapshot& destination) const noexcept;
  void set_servo_gains(std::span<const float> stiffness,
                       std::span<const float> damping) noexcept;
  void physics_loop() noexcept;
  bool configure_physics_thread() noexcept;
  void publish_low_state() noexcept;

  std::unique_ptr<Impl> impl_;
  std::unique_ptr<CommandSlot> command_slot_;
  std::array<float, kJointCount> default_joint_position_{};
  std::array<float, kJointCount> effort_limit_{};
  std::array<float, kJointCount> hold_stiffness_{};
  std::array<float, kJointCount> hold_damping_{};
  std::array<float, 36> initial_pose_{};
  bool has_initial_pose_ = false;
  std::array<std::size_t, kJointCount> actuator_to_sdk_{};
  std::array<int, kJointCount> sdk_to_actuator_{};
  std::array<int, kJointCount> qpos_address_{};
  std::array<int, kJointCount> dof_address_{};
  int pelvis_body_id_ = -1;
  double timestep_ = 0.002;
  std::uint8_t mode_machine_ = 5;
  int physics_cpu_ = -1;
  int physics_fifo_priority_ = 0;
  bool lock_memory_ = false;
  bool require_realtime_ = false;

  std::thread physics_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> thread_ready_{false};
  std::atomic<bool> realtime_configured_{false};
  std::atomic<bool> physics_fault_{false};
  std::atomic<bool> holding_{true};
  std::atomic<std::uint64_t> steps_{0};
  std::atomic<std::uint64_t> publishes_{0};
  std::atomic<std::uint64_t> publish_failures_{0};
  std::atomic<std::uint64_t> commands_received_{0};
  std::atomic<std::uint64_t> crc_errors_{0};
  std::atomic<std::uint64_t> wake_late_ns_max_{0};
  std::atomic<std::uint64_t> deadline_misses_{0};
  std::atomic<std::uint64_t> last_command_ns_{0};
  std::atomic<double> simulation_time_{0.0};
  std::atomic<float> base_height_{0.0F};
  std::atomic<float> min_base_height_{0.0F};
  std::atomic<float> applied_kp_min_{-1.0F};
  std::atomic<float> applied_kp_max_{-1.0F};
  std::atomic<float> applied_kd_min_{-1.0F};
  std::atomic<float> applied_kd_max_{-1.0F};
  std::atomic<float> applied_q_absmax_{-1.0F};
  std::atomic<float> applied_extra_absmax_{-1.0F};
};

}  // namespace ec_native
