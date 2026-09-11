#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "onnx_engine.hpp"

namespace ec_native {

constexpr std::size_t kJointCount = 29;
constexpr std::size_t kMaxObservation = 1024;
constexpr std::size_t kMaxCommand = 1024;

enum class TermKind {
  kCommand,
  kProjectedGravity,
  kBaseAngularVelocity,
  kJointPositionRelative,
  kJointVelocityRelative,
  kLastAction,
};

enum class HistoryOrder {
  kOldestFirst,
  kNewestFirst,
};

enum class ResetFill {
  kRepeatFirst,
  kZero,
};

struct TermConfig {
  std::string name;
  std::size_t width;
  std::size_t history_length;
  std::size_t history_stride;
  HistoryOrder history_order;
  ResetFill reset_fill;
};

struct TermSpec {
  TermKind kind;
  std::size_t sample_width;
  std::size_t history_length;
  std::size_t history_stride;
  std::size_t history_span;
  HistoryOrder history_order;
  ResetFill reset_fill;
  std::size_t observation_offset;
  std::size_t command_offset;
  std::size_t history_offset;
  std::size_t history_cursor = 0;
  bool history_initialized = false;
};

struct RobotState {
  std::array<float, kJointCount> joint_position{};
  std::array<float, kJointCount> joint_velocity{};
  std::array<float, 3> projected_gravity{};
  std::array<float, 3> base_angular_velocity{};
  std::array<float, 3> anchor_position_w{};
  std::array<float, 4> anchor_quaternion_w{0.0F, 0.0F, 0.0F, 1.0F};
  bool anchor_pose_valid = false;
};

bool align_heading_to_reference(
    std::span<const float> initial_robot_quaternion,
    std::span<const float> initial_reference_quaternion,
    std::span<const float> robot_quaternion,
    std::span<float> aligned_quaternion) noexcept;

// Convert raw world-frame reference records
// [joint qpos 29 | anchor position 3 | anchor quaternion XYZW 4] into
// robot-anchored root_qpos frames
// [joint qpos 29 | relative position 3 | relative orientation rot6d 6].
// This is allocation-free and matches lowlevel.maths.subtract_frame.
bool reexpress_root_qpos_window(
    std::span<const float> raw_world_frames, std::size_t frame_count,
    std::span<const float> anchor_position_w,
    std::span<const float> anchor_quaternion_w,
    std::span<float> root_qpos_frames, bool heading_only = false) noexcept;

bool pack_joint_qpos_qvel_anchor_ori_window(
    std::span<const float> raw_world_frames, std::size_t available_frames,
    std::size_t start_frame, std::size_t frame_count,
    std::size_t frame_stride, std::span<const float> robot_quaternion_w,
    std::span<float> encoder_window) noexcept;

struct StepResult {
  std::array<float, kMaxObservation> observation{};
  std::size_t observation_width = 0;
  std::array<float, kJointCount> action{};
  std::array<float, kJointCount> joint_target{};
};

class NativeTrackerCore {
 public:
  NativeTrackerCore(
      const std::string& policy_path, const std::string& input_name,
      const std::string& output_name,
      const std::vector<TermConfig>& terms,
      std::size_t command_width,
      std::span<const float> default_joint_position,
      std::span<const float> action_scale,
      std::span<const float> joint_lower, std::span<const float> joint_upper,
      std::span<const float> fsq_half_levels, std::size_t fsq_z_dim,
      float raw_action_clip,
      std::size_t intra_op_threads = 1);

  void reset() noexcept;
  const StepResult& step(const RobotState& state,
                         std::span<const float> command);
  void warmup(std::size_t iterations = 8);

  std::size_t observation_width() const noexcept { return observation_width_; }
  std::size_t command_width() const noexcept { return command_width_; }
  const std::array<float, kJointCount>& default_joint_position() const noexcept {
    return default_joint_position_;
  }
  const std::array<float, kJointCount>& last_action() const noexcept {
    return last_action_;
  }
  // Overwrite the action the next observation reports as "last": what the
  // writer actually applied, when that differs from what the policy asked
  // (the blend-in). Isaac's last_action is the executed action; so is this.
  void set_last_action(std::span<const float> action) noexcept;

 private:
  static TermKind parse_term(const std::string& name);
  void validate_state(const RobotState& state) const;
  void update_history(TermSpec& term, const RobotState& state,
                      std::span<const float> command);
  void assemble(const RobotState& state, std::span<const float> command);

  std::vector<TermSpec> terms_;
  std::size_t observation_width_ = 0;
  std::size_t command_width_ = 0;
  std::array<float, kJointCount> default_joint_position_{};
  std::array<float, kJointCount> action_scale_{};
  std::array<float, kJointCount> joint_lower_{};
  std::array<float, kJointCount> joint_upper_{};
  std::vector<float> fsq_half_levels_;
  std::size_t fsq_z_dim_ = 0;
  float raw_action_clip_ = 0.0F;
  std::array<float, kJointCount> last_action_{};
  std::vector<float> history_storage_;
  StepResult result_{};
  OnnxEngine engine_;
};

}  // namespace ec_native
