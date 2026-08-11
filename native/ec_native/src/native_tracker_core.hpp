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

struct TermSpec {
  TermKind kind;
  std::size_t width;
  std::size_t observation_offset;
  std::size_t command_offset;
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

// Convert raw world-frame reference records
// [joint qpos 29 | anchor position 3 | anchor quaternion XYZW 4] into
// robot-anchored root_qpos frames
// [joint qpos 29 | relative position 3 | relative orientation rot6d 6].
// This is allocation-free and matches lowlevel.maths.subtract_frame.
bool reexpress_root_qpos_window(
    std::span<const float> raw_world_frames, std::size_t frame_count,
    std::span<const float> anchor_position_w,
    std::span<const float> anchor_quaternion_w,
    std::span<float> root_qpos_frames) noexcept;

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
      const std::vector<std::pair<std::string, std::size_t>>& terms,
      std::size_t command_width,
      std::span<const float> default_joint_position,
      std::span<const float> action_scale,
      std::span<const float> joint_lower, std::span<const float> joint_upper,
      std::span<const float> fsq_half_levels, std::size_t fsq_z_dim,
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

 private:
  static TermKind parse_term(const std::string& name);
  void validate_state(const RobotState& state) const;
  void assemble(const RobotState& state, std::span<const float> command);

  std::vector<TermSpec> terms_;
  std::size_t observation_width_ = 0;
  std::size_t command_width_ = 0;
  std::array<float, kJointCount> default_joint_position_{};
  std::array<float, kJointCount> action_scale_{};
  std::array<float, kJointCount> joint_lower_{};
  std::array<float, kJointCount> joint_upper_{};
  bool clamp_joint_targets_ = false;
  std::vector<float> fsq_half_levels_;
  std::size_t fsq_z_dim_ = 0;
  std::array<float, kJointCount> last_action_{};
  StepResult result_{};
  OnnxEngine engine_;
};

}  // namespace ec_native
