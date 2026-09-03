#include "native_tracker_core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ec_native {
namespace {

constexpr std::size_t kRawReferenceWidth = kJointCount + 3 + 4;
constexpr std::size_t kRootQposWidth = kJointCount + 3 + 6;
constexpr std::size_t kJointReferenceRawWidth = kJointCount * 2 + 4;
constexpr std::size_t kJointReferenceFrameWidth = kJointCount * 2 + 6;

bool normalized_quaternion(std::span<const float> input,
                           std::array<float, 4>& output) noexcept {
  if (input.size() != 4) {
    return false;
  }
  double norm_squared = 0.0;
  for (float value : input) {
    if (!std::isfinite(value)) {
      return false;
    }
    norm_squared += static_cast<double>(value) * value;
  }
  if (norm_squared <= 1.0e-12) {
    return false;
  }
  const float inverse_norm =
      static_cast<float>(1.0 / std::sqrt(norm_squared));
  for (std::size_t index = 0; index < 4; ++index) {
    output[index] = input[index] * inverse_norm;
  }
  return true;
}

std::array<float, 4> multiply_quaternions(
    const std::array<float, 4>& a,
    const std::array<float, 4>& b) noexcept {
  const float ax = a[0];
  const float ay = a[1];
  const float az = a[2];
  const float aw = a[3];
  const float bx = b[0];
  const float by = b[1];
  const float bz = b[2];
  const float bw = b[3];
  return {
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
      aw * bw - ax * bx - ay * by - az * bz,
  };
}

std::array<float, 9> quaternion_matrix(
    const std::array<float, 4>& q) noexcept {
  const float x = q[0];
  const float y = q[1];
  const float z = q[2];
  const float w = q[3];
  return {
      1.0F - 2.0F * (y * y + z * z),
      2.0F * (x * y - z * w),
      2.0F * (x * z + y * w),
      2.0F * (x * y + z * w),
      1.0F - 2.0F * (x * x + z * z),
      2.0F * (y * z - x * w),
      2.0F * (x * z - y * w),
      2.0F * (y * z + x * w),
      1.0F - 2.0F * (x * x + y * y),
  };
}

bool is_command_term(const std::string& name) {
  return name == "latent_command" || name == "expert_motion" ||
         name == "expert_anchor_pos_b" || name == "expert_anchor_ori_b";
}

template <std::size_t Size>
bool all_finite(const std::array<float, Size>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); });
}

bool all_finite(std::span<const float> values) {
  return std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); });
}

std::size_t expected_width(TermKind kind) {
  switch (kind) {
    case TermKind::kCommand:
      return 0;
    case TermKind::kProjectedGravity:
    case TermKind::kBaseAngularVelocity:
      return 3;
    case TermKind::kJointPositionRelative:
    case TermKind::kJointVelocityRelative:
    case TermKind::kLastAction:
      return kJointCount;
  }
  return 0;
}

}  // namespace

bool align_heading_to_reference(
    std::span<const float> initial_robot_quaternion,
    std::span<const float> initial_reference_quaternion,
    std::span<const float> robot_quaternion,
    std::span<float> aligned_quaternion) noexcept {
  std::array<float, 4> initial{}, reference{}, current{};
  if (aligned_quaternion.size() != 4 ||
      !normalized_quaternion(initial_robot_quaternion, initial) ||
      !normalized_quaternion(initial_reference_quaternion, reference) ||
      !normalized_quaternion(robot_quaternion, current)) {
    return false;
  }
  initial[0] = initial[1] = reference[0] = reference[1] = 0.0F;
  if (!normalized_quaternion(initial, initial) ||
      !normalized_quaternion(reference, reference)) {
    return false;
  }
  initial[2] = -initial[2];
  // Only world yaw is arbitrary. Cancelling the full initial pose would
  // erase real tilt error and disagree with the policy's projected gravity.
  const auto offset = multiply_quaternions(reference, initial);
  const auto aligned = multiply_quaternions(offset, current);
  std::copy(aligned.begin(), aligned.end(), aligned_quaternion.begin());
  return true;
}

bool reexpress_root_qpos_window(
    std::span<const float> raw_world_frames, std::size_t frame_count,
    std::span<const float> anchor_position_w,
    std::span<const float> anchor_quaternion_w,
    std::span<float> root_qpos_frames) noexcept {
  if (frame_count == 0 ||
      raw_world_frames.size() != frame_count * kRawReferenceWidth ||
      root_qpos_frames.size() != frame_count * kRootQposWidth ||
      anchor_position_w.size() != 3 || anchor_quaternion_w.size() != 4 ||
      !std::all_of(anchor_position_w.begin(), anchor_position_w.end(),
                   [](float value) { return std::isfinite(value); })) {
    return false;
  }
  std::array<float, 4> robot_quaternion{};
  if (!normalized_quaternion(anchor_quaternion_w, robot_quaternion)) {
    return false;
  }
  const auto robot_matrix = quaternion_matrix(robot_quaternion);
  const std::array<float, 4> robot_conjugate = {
      -robot_quaternion[0], -robot_quaternion[1], -robot_quaternion[2],
      robot_quaternion[3]};

  for (std::size_t frame = 0; frame < frame_count; ++frame) {
    const float* source =
        raw_world_frames.data() + frame * kRawReferenceWidth;
    float* destination = root_qpos_frames.data() + frame * kRootQposWidth;
    if (!std::all_of(source, source + kRawReferenceWidth,
                     [](float value) { return std::isfinite(value); })) {
      return false;
    }
    std::copy_n(source, kJointCount, destination);
    const float dx = source[kJointCount] - anchor_position_w[0];
    const float dy = source[kJointCount + 1] - anchor_position_w[1];
    const float dz = source[kJointCount + 2] - anchor_position_w[2];
    destination[kJointCount] =
        robot_matrix[0] * dx + robot_matrix[3] * dy + robot_matrix[6] * dz;
    destination[kJointCount + 1] =
        robot_matrix[1] * dx + robot_matrix[4] * dy + robot_matrix[7] * dz;
    destination[kJointCount + 2] =
        robot_matrix[2] * dx + robot_matrix[5] * dy + robot_matrix[8] * dz;

    std::array<float, 4> reference_quaternion{};
    if (!normalized_quaternion(
            std::span<const float>(source + kJointCount + 3, 4),
            reference_quaternion)) {
      return false;
    }
    std::array<float, 4> relative_quaternion =
        multiply_quaternions(robot_conjugate, reference_quaternion);
    if (!normalized_quaternion(relative_quaternion, relative_quaternion)) {
      return false;
    }
    const auto relative_matrix = quaternion_matrix(relative_quaternion);
    float* orientation = destination + kJointCount + 3;
    orientation[0] = relative_matrix[0];
    orientation[1] = relative_matrix[1];
    orientation[2] = relative_matrix[3];
    orientation[3] = relative_matrix[4];
    orientation[4] = relative_matrix[6];
    orientation[5] = relative_matrix[7];
  }
  return true;
}

bool pack_joint_qpos_qvel_anchor_ori_window(
    std::span<const float> raw_world_frames, std::size_t available_frames,
    std::size_t start_frame, std::size_t frame_count,
    std::size_t frame_stride, std::span<const float> robot_quaternion_w,
    std::span<float> encoder_window) noexcept {
  if (frame_count == 0 || frame_count % 2 != 0 || frame_stride == 0 ||
      raw_world_frames.size() != available_frames * kJointReferenceRawWidth ||
      encoder_window.size() != frame_count * kJointReferenceFrameWidth ||
      start_frame + (frame_count - 1) * frame_stride >= available_frames) {
    return false;
  }
  std::array<float, 4> robot_quaternion{};
  if (!normalized_quaternion(robot_quaternion_w, robot_quaternion)) {
    return false;
  }
  std::array<float, 4> heading = {
      0.0F, 0.0F, robot_quaternion[2], robot_quaternion[3]};
  if (!normalized_quaternion(heading, heading)) {
    return false;
  }
  const std::array<float, 4> heading_conjugate = {
      -heading[0], -heading[1], -heading[2], heading[3]};
  std::array<std::array<float, 6>, 32> orientations{};
  if (frame_count > orientations.size()) {
    return false;
  }
  for (std::size_t frame = 0; frame < frame_count; ++frame) {
    const float* source = raw_world_frames.data() +
                          (start_frame + frame * frame_stride) *
                              kJointReferenceRawWidth;
    if (!std::all_of(source, source + kJointReferenceRawWidth,
                     [](float value) { return std::isfinite(value); })) {
      return false;
    }
    std::array<float, 4> reference_quaternion{};
    if (!normalized_quaternion(
            std::span<const float>(source + kJointCount * 2, 4),
            reference_quaternion)) {
      return false;
    }
    std::array<float, 4> relative =
        multiply_quaternions(heading_conjugate, reference_quaternion);
    if (!normalized_quaternion(relative, relative)) {
      return false;
    }
    const auto matrix = quaternion_matrix(relative);
    orientations[frame] = {
        matrix[0], matrix[1], matrix[3],
        matrix[4], matrix[6], matrix[7],
    };
  }

  float* destination = encoder_window.data();
  const std::size_t half = frame_count / 2;
  for (std::size_t block = 0; block < half; ++block) {
    for (std::size_t pair = 0; pair < 2; ++pair) {
      const std::size_t frame = block * 2 + pair;
      const float* source = raw_world_frames.data() +
                            (start_frame + frame * frame_stride) *
                                kJointReferenceRawWidth;
      std::copy_n(source, kJointCount, destination);
      destination += kJointCount;
    }
    std::copy(orientations[block].begin(), orientations[block].end(),
              destination);
    destination += 6;
  }
  for (std::size_t block = 0; block < half; ++block) {
    for (std::size_t pair = 0; pair < 2; ++pair) {
      const std::size_t frame = block * 2 + pair;
      const float* source = raw_world_frames.data() +
                            (start_frame + frame * frame_stride) *
                                kJointReferenceRawWidth + kJointCount;
      std::copy_n(source, kJointCount, destination);
      destination += kJointCount;
    }
    std::copy(orientations[half + block].begin(),
              orientations[half + block].end(), destination);
    destination += 6;
  }
  return destination == encoder_window.data() + encoder_window.size();
}

TermKind NativeTrackerCore::parse_term(const std::string& name) {
  if (is_command_term(name)) {
    return TermKind::kCommand;
  }
  if (name == "projected_gravity") {
    return TermKind::kProjectedGravity;
  }
  if (name == "base_ang_vel") {
    return TermKind::kBaseAngularVelocity;
  }
  if (name == "joint_pos_rel") {
    return TermKind::kJointPositionRelative;
  }
  if (name == "joint_vel_rel") {
    return TermKind::kJointVelocityRelative;
  }
  if (name == "last_action") {
    return TermKind::kLastAction;
  }
  throw std::runtime_error("unsupported observation term: " + name);
}

NativeTrackerCore::NativeTrackerCore(
    const std::string& policy_path, const std::string& input_name,
    const std::string& output_name,
    const std::vector<TermConfig>& terms,
    std::size_t command_width,
    std::span<const float> default_joint_position,
    std::span<const float> action_scale,
    std::span<const float> joint_lower, std::span<const float> joint_upper,
    std::span<const float> fsq_half_levels, std::size_t fsq_z_dim,
    float raw_action_clip,
    std::size_t intra_op_threads)
    : command_width_(command_width),
      fsq_half_levels_(fsq_half_levels.begin(), fsq_half_levels.end()),
      fsq_z_dim_(fsq_z_dim),
      raw_action_clip_(raw_action_clip),
      engine_(policy_path, input_name, output_name,
              [&terms]() {
                std::size_t total = 0;
                for (const auto& term : terms) {
                  total += term.width * term.history_length;
                }
                return total;
              }(),
              kJointCount, intra_op_threads) {
  if (command_width_ == 0 || command_width_ > kMaxCommand) {
    throw std::runtime_error("command width is outside native limits");
  }
  if (!std::isfinite(raw_action_clip_) || raw_action_clip_ < 0.0F) {
    throw std::runtime_error("raw action clip must be finite and non-negative");
  }
  if (default_joint_position.size() != kJointCount ||
      action_scale.size() != kJointCount ||
      !all_finite(default_joint_position) || !all_finite(action_scale)) {
    throw std::runtime_error(
        "default joint position and action scale must have width 29");
  }
  std::copy(default_joint_position.begin(), default_joint_position.end(),
            default_joint_position_.begin());
  std::copy(action_scale.begin(), action_scale.end(), action_scale_.begin());
  if (!joint_lower.empty() || !joint_upper.empty()) {
    if (joint_lower.size() != kJointCount ||
        joint_upper.size() != kJointCount || !all_finite(joint_lower) ||
        !all_finite(joint_upper)) {
      throw std::runtime_error("joint-limit arrays must both have width 29");
    }
    for (std::size_t index = 0; index < kJointCount; ++index) {
      if (joint_lower[index] > joint_upper[index]) {
        throw std::runtime_error("joint lower limit exceeds upper limit");
      }
    }
    std::copy(joint_lower.begin(), joint_lower.end(), joint_lower_.begin());
    std::copy(joint_upper.begin(), joint_upper.end(), joint_upper_.begin());
    clamp_joint_targets_ = true;
  }
  if (fsq_z_dim_ > command_width_ ||
      fsq_half_levels_.size() != fsq_z_dim_ ||
      !all_finite(fsq_half_levels_) ||
      std::any_of(fsq_half_levels_.begin(), fsq_half_levels_.end(),
                  [](float value) { return value <= 0.0F; })) {
    throw std::runtime_error("invalid FSQ contract");
  }

  std::size_t observation_offset = 0;
  std::size_t command_offset = 0;
  std::size_t history_offset = 0;
  for (const auto& config : terms) {
    const auto kind = parse_term(config.name);
    if (config.width == 0 || config.history_length == 0 ||
        config.history_stride == 0) {
      throw std::runtime_error("observation history fields must be positive");
    }
    const std::size_t flat_width = config.width * config.history_length;
    const std::size_t history_span =
        1 + (config.history_length - 1) * config.history_stride;
    const std::size_t history_values = config.width * history_span;
    if (flat_width / config.history_length != config.width ||
        history_values / history_span != config.width ||
        observation_offset + flat_width > kMaxObservation) {
      throw std::runtime_error("observation term exceeds native limits");
    }
    const std::size_t required_width = expected_width(kind);
    if (required_width != 0 && config.width != required_width) {
      throw std::runtime_error("observation term has an invalid width: " +
                               config.name);
    }
    const auto source_offset = kind == TermKind::kCommand ? command_offset : 0;
    terms_.push_back({kind,
                      config.width,
                      config.history_length,
                      config.history_stride,
                      history_span,
                      config.history_order,
                      config.reset_fill,
                      observation_offset,
                      source_offset,
                      history_offset});
    observation_offset += flat_width;
    history_offset += history_values;
    if (kind == TermKind::kCommand) {
      command_offset += config.width;
    }
  }
  history_storage_.resize(history_offset);
  observation_width_ = observation_offset;
  result_.observation_width = observation_width_;
  if (command_offset != command_width_) {
    throw std::runtime_error("command terms do not total the command width");
  }
  reset();
}

void NativeTrackerCore::reset() noexcept {
  last_action_.fill(0.0F);
  std::fill(history_storage_.begin(), history_storage_.end(), 0.0F);
  for (auto& term : terms_) {
    term.history_cursor = 0;
    term.history_initialized = false;
  }
}

void NativeTrackerCore::validate_state(const RobotState& state) const {
  if (!all_finite(state.joint_position) ||
      !all_finite(state.joint_velocity) ||
      !all_finite(state.projected_gravity) ||
      !all_finite(state.base_angular_velocity)) {
    throw std::runtime_error("robot state contains a non-finite value");
  }
}

void NativeTrackerCore::update_history(TermSpec& term, const RobotState& state,
                                       std::span<const float> command) {
  if (term.history_initialized) {
    term.history_cursor = (term.history_cursor + 1) % term.history_span;
  }
  float* sample = history_storage_.data() + term.history_offset +
                  term.history_cursor * term.sample_width;
  switch (term.kind) {
    case TermKind::kCommand:
      for (std::size_t index = 0; index < term.sample_width; ++index) {
        const std::size_t command_index = term.command_offset + index;
        float value = command[command_index];
        if (command_index < fsq_z_dim_) {
          const float half = fsq_half_levels_[command_index];
          value = std::clamp(std::nearbyint(value * half), -half,
                             half - 1.0F) /
                  half;
        }
        sample[index] = value;
      }
      break;
    case TermKind::kProjectedGravity:
      std::copy_n(state.projected_gravity.begin(), term.sample_width, sample);
      break;
    case TermKind::kBaseAngularVelocity:
      std::copy_n(state.base_angular_velocity.begin(), term.sample_width,
                  sample);
      break;
    case TermKind::kJointPositionRelative:
      for (std::size_t index = 0; index < term.sample_width; ++index) {
        sample[index] =
            state.joint_position[index] - default_joint_position_[index];
      }
      break;
    case TermKind::kJointVelocityRelative:
      std::copy_n(state.joint_velocity.begin(), term.sample_width, sample);
      break;
    case TermKind::kLastAction:
      std::copy_n(last_action_.begin(), term.sample_width, sample);
      break;
  }
  if (!term.history_initialized && term.reset_fill == ResetFill::kRepeatFirst) {
    for (std::size_t slot = 1; slot < term.history_span; ++slot) {
      std::copy_n(sample, term.sample_width,
                  history_storage_.data() + term.history_offset +
                      slot * term.sample_width);
    }
  }
  term.history_initialized = true;
}

void NativeTrackerCore::assemble(const RobotState& state,
                                 std::span<const float> command) {
  if (command.size() != command_width_) {
    throw std::runtime_error("command width mismatch");
  }
  if (!std::all_of(command.begin(), command.end(),
                   [](float value) { return std::isfinite(value); })) {
    throw std::runtime_error("command contains a non-finite value");
  }
  for (auto& term : terms_) {
    update_history(term, state, command);
    float* destination = result_.observation.data() + term.observation_offset;
    for (std::size_t output_frame = 0;
         output_frame < term.history_length; ++output_frame) {
      const std::size_t history_index =
          term.history_order == HistoryOrder::kOldestFirst
              ? term.history_length - 1 - output_frame
              : output_frame;
      const std::size_t age = history_index * term.history_stride;
      const std::size_t source_slot =
          (term.history_cursor + term.history_span - age) % term.history_span;
      std::copy_n(history_storage_.data() + term.history_offset +
                      source_slot * term.sample_width,
                  term.sample_width,
                  destination + output_frame * term.sample_width);
    }
  }
}

const StepResult& NativeTrackerCore::step(const RobotState& state,
                                          std::span<const float> command) {
  validate_state(state);
  assemble(state, command);
  const auto action = engine_.infer(
      std::span<const float>(result_.observation.data(), observation_width_));
  for (std::size_t index = 0; index < kJointCount; ++index) {
    if (!std::isfinite(action[index])) {
      throw std::runtime_error("policy action contains a non-finite value");
    }
    const float clipped_action =
        raw_action_clip_ > 0.0F
            ? std::clamp(action[index], -raw_action_clip_, raw_action_clip_)
            : action[index];
    result_.action[index] = clipped_action;
    float target = default_joint_position_[index] +
                   action_scale_[index] * clipped_action;
    if (clamp_joint_targets_) {
      target = std::clamp(target, joint_lower_[index], joint_upper_[index]);
    }
    result_.joint_target[index] = target;
    last_action_[index] = clipped_action;
  }
  return result_;
}

void NativeTrackerCore::set_last_action(std::span<const float> action) noexcept {
  if (action.size() != kJointCount) {
    return;
  }
  for (std::size_t index = 0; index < kJointCount; ++index) {
    const float value = std::isfinite(action[index]) ? action[index] : 0.0F;
    last_action_[index] =
        raw_action_clip_ > 0.0F
            ? std::clamp(value, -raw_action_clip_, raw_action_clip_)
            : value;
  }
}

void NativeTrackerCore::warmup(std::size_t iterations) {
  engine_.warmup(iterations);
}

}  // namespace ec_native
