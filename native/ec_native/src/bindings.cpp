#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "native_tracker_core.hpp"
#include "leg_odometry.hpp"
#include "native_fake_runtime.hpp"
#include "shm_command_slot.hpp"
#ifdef EC_WITH_UNITREE
#include "g1_loco_client.hpp"
#include "mujoco_dds_plant.hpp"
#include "plant_vendor.hpp"
#include "unitree_backend.hpp"
#include "unitree_state_probe.hpp"
#include "odometry_probe.hpp"
#endif

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

std::vector<float> vector_from_array(const FloatArray& values,
                                     std::size_t expected,
                                     const char* name,
                                     bool allow_empty = false) {
  const auto info = values.request();
  if (info.ndim != 1 ||
      ((!allow_empty || info.shape[0] != 0) &&
       info.shape[0] != static_cast<py::ssize_t>(expected))) {
    throw std::runtime_error(std::string(name) + " must have width " +
                             std::to_string(expected));
  }
  const auto* begin = static_cast<const float*>(info.ptr);
  return {begin, begin + info.shape[0]};
}

template <std::size_t Size>
void copy_array(const FloatArray& source, std::array<float, Size>& destination,
                const char* name) {
  const auto values = vector_from_array(source, Size, name);
  std::copy(values.begin(), values.end(), destination.begin());
}

py::array_t<float> reexpress_root_qpos_binding(
    const FloatArray& raw_world_frames, const FloatArray& anchor_position_w,
    const FloatArray& anchor_quaternion_w, bool heading_only) {
  const auto raw = raw_world_frames.request();
  if (raw.ndim != 2 || raw.shape[1] != 36 || raw.shape[0] <= 0) {
    throw std::runtime_error("raw_world_frames must have shape [H, 36]");
  }
  const auto position = vector_from_array(anchor_position_w, 3,
                                          "anchor_position_w");
  const auto quaternion = vector_from_array(anchor_quaternion_w, 4,
                                            "anchor_quaternion_w");
  py::array_t<float> output({raw.shape[0], static_cast<py::ssize_t>(38)});
  const auto* raw_data = static_cast<const float*>(raw.ptr);
  if (!ec_native::reexpress_root_qpos_window(
          std::span<const float>(raw_data,
                                 static_cast<std::size_t>(raw.shape[0]) * 36),
          static_cast<std::size_t>(raw.shape[0]), position, quaternion,
          std::span<float>(output.mutable_data(),
                           static_cast<std::size_t>(raw.shape[0]) * 38), heading_only)) {
    throw std::runtime_error("raw reference pose is invalid");
  }
  return output;
}

py::array_t<float> align_heading_binding(
    const FloatArray& initial_robot, const FloatArray& initial_reference,
    const FloatArray& current_robot) {
  const auto initial = vector_from_array(initial_robot, 4, "initial_robot");
  const auto reference = vector_from_array(initial_reference, 4, "initial_reference");
  const auto current = vector_from_array(current_robot, 4, "current_robot");
  py::array_t<float> output(4);
  if (!ec_native::align_heading_to_reference(
          initial, reference, current,
          std::span<float>(output.mutable_data(), 4))) {
    throw std::runtime_error("heading alignment quaternion is invalid");
  }
  return output;
}

py::array_t<float> projected_gravity_binding(const FloatArray& quaternion) {
  const auto values = vector_from_array(quaternion, 4, "quaternion_xyzw");
  py::array_t<float> output(3);
  if (!ec_native::projected_gravity_from_xyzw(
          values, std::span<float>(output.mutable_data(), 3))) {
    throw std::runtime_error("quaternion_xyzw is invalid");
  }
  return output;
}

py::array_t<float> pack_joint_reference_binding(
    const FloatArray& raw_world_frames, std::size_t start_frame,
    std::size_t frame_count, std::size_t frame_stride,
    const FloatArray& robot_quaternion_w) {
  const auto raw = raw_world_frames.request();
  if (raw.ndim != 2 || raw.shape[1] != 62 || raw.shape[0] <= 0) {
    throw std::runtime_error("raw_world_frames must have shape [H, 62]");
  }
  const auto quaternion =
      vector_from_array(robot_quaternion_w, 4, "robot_quaternion_w");
  py::array_t<float> output(
      static_cast<py::ssize_t>(frame_count * 64));
  const auto* raw_data = static_cast<const float*>(raw.ptr);
  if (!ec_native::pack_joint_qpos_qvel_anchor_ori_window(
          std::span<const float>(raw_data,
                                 static_cast<std::size_t>(raw.shape[0]) * 62),
          static_cast<std::size_t>(raw.shape[0]), start_frame, frame_count,
          frame_stride, quaternion,
          std::span<float>(output.mutable_data(), frame_count * 64))) {
    throw std::runtime_error("joint reference window is invalid");
  }
  return output;
}

ec_native::NativePlannerConfig::ReferenceEncoderLayout reference_layout(
    const std::string& value) {
  if (value == "root_qpos_heading") {
    return ec_native::NativePlannerConfig::ReferenceEncoderLayout::kRootQposHeading;
  }
  if (value == "root_qpos") {
    return ec_native::NativePlannerConfig::ReferenceEncoderLayout::kRootQpos;
  }
  if (value == "joint_qpos_qvel_anchor_ori") {
    return ec_native::NativePlannerConfig::ReferenceEncoderLayout::
        kJointQposQvelAnchorOri;
  }
  throw std::runtime_error("unsupported reference encoder layout: " + value);
}

ec_native::NativePlannerConfig::EncoderTrigger encoder_trigger(
    const std::string& value) {
  if (value == "on_acceptance") {
    return ec_native::NativePlannerConfig::EncoderTrigger::kOnAcceptance;
  }
  if (value == "every_control_tick") {
    return ec_native::NativePlannerConfig::EncoderTrigger::kEveryControlTick;
  }
  throw std::runtime_error("unsupported encoder trigger: " + value);
}

ec_native::NativePlannerConfig::AnchorSource anchor_source(
    const std::string& value) {
  if (value == "robot") {
    return ec_native::NativePlannerConfig::AnchorSource::kRobot;
  }
  if (value == "expert_heading") {
    return ec_native::NativePlannerConfig::AnchorSource::kExpertHeading;
  }
  throw std::runtime_error("unsupported anchor source: " + value);
}

#ifdef EC_WITH_UNITREE
ec_native::AnchorPositionSource anchor_position_source(
    const std::string& value) {
  if (value == "fixed_start") {
    return ec_native::AnchorPositionSource::kFixedStart;
  }
  if (value == "odometry") {
    return ec_native::AnchorPositionSource::kOdometry;
  }
  if (value == "leg_kinematics") {
    return ec_native::AnchorPositionSource::kLegKinematics;
  }
  if (value == "auto") {
    return ec_native::AnchorPositionSource::kAuto;
  }
  throw std::runtime_error("unsupported anchor position source: " + value);
}
#endif

class LegOdometryBinding {
 public:
  LegOdometryBinding(const std::string& mjcf_path,
                     const std::vector<std::string>& isaac_joint_names,
                     float contact_threshold_m)
      : odometry_(mjcf_path, isaac_joint_names, contact_threshold_m) {}

  void reset() { odometry_.reset(); }

  py::array_t<float> update(const FloatArray& joint_position,
                            const FloatArray& quaternion_xyzw) {
    const auto q = vector_from_array(joint_position, ec_native::kJointCount,
                                     "joint_position");
    const auto quaternion =
        vector_from_array(quaternion_xyzw, 4, "quaternion_xyzw");
    std::array<float, 3> position{};
    if (!odometry_.update(q, quaternion, position)) {
      throw std::runtime_error("leg odometry rejected a non-finite input");
    }
    py::array_t<float> out(3);
    std::copy(position.begin(), position.end(), out.mutable_data());
    return out;
  }

  int stance_foot() const { return odometry_.stance_foot(); }
  std::uint64_t stance_switches() const { return odometry_.stance_switches(); }

 private:
  ec_native::LegOdometry odometry_;
};

class ShmCommandSlot {
 public:
  ShmCommandSlot(const std::string& name, bool create)
      : slot_(name, create) {}

  void publish(std::uint64_t sequence, std::uint32_t interface_tag,
               py::array_t<float, py::array::c_style | py::array::forcecast> values,
               double sender_stamp) {
    const auto info = values.request();
    if (info.ndim != 1) {
      throw std::runtime_error("values must be a 1-D float32 array");
    }
    const auto* data = static_cast<const float*>(info.ptr);
    const auto length = static_cast<std::uint32_t>(info.shape[0]);
    py::gil_scoped_release release;
    slot_.publish(sequence, interface_tag, data, length, sender_stamp);
  }

  py::object snapshot(std::uint64_t after_sequence) {
    std::uint64_t sequence = 0;
    std::uint32_t tag = 0;
    std::uint32_t length = 0;
    double recv_stamp = 0.0;
    double sender_stamp = 0.0;
    static thread_local float scratch[ec_native::kMaxValues];
    bool ok = false;
    {
      py::gil_scoped_release release;
      ok = slot_.snapshot(sequence, tag, length, recv_stamp, sender_stamp,
                          scratch, ec_native::kMaxValues, after_sequence);
    }
    if (!ok) {
      return py::none();
    }
    py::array_t<float> values(static_cast<py::ssize_t>(length));
    std::memcpy(values.mutable_data(), scratch, sizeof(float) * length);
    return py::make_tuple(sequence, tag, values, recv_stamp, sender_stamp);
  }

 private:
  ec_native::ShmSlot slot_;
};

class NativeTrackerCoreBinding {
 public:
  using TermTuple = std::tuple<std::string, std::size_t, std::size_t,
                               std::size_t, std::string, std::string>;

  NativeTrackerCoreBinding(
      const std::string& policy_path, const std::string& input_name,
      const std::string& output_name,
      const std::vector<TermTuple>& terms,
      std::size_t command_width, const FloatArray& default_joint_position,
      const FloatArray& action_scale, const FloatArray& joint_lower,
      const FloatArray& joint_upper, const FloatArray& fsq_half_levels,
      std::size_t fsq_z_dim, float raw_action_clip,
      std::size_t intra_op_threads) {
    const auto defaults = vector_from_array(
        default_joint_position, ec_native::kJointCount, "default_joint_position");
    const auto scales =
        vector_from_array(action_scale, ec_native::kJointCount, "action_scale");
    const auto lower = vector_from_array(
        joint_lower, ec_native::kJointCount, "joint_lower", true);
    const auto upper = vector_from_array(
        joint_upper, ec_native::kJointCount, "joint_upper", true);
    const auto half = vector_from_array(fsq_half_levels, fsq_z_dim,
                                        "fsq_half_levels", true);
    std::vector<ec_native::TermConfig> term_configs;
    term_configs.reserve(terms.size());
    for (const auto& [name, width, history_length, history_stride,
                      history_order, reset_fill] : terms) {
      ec_native::HistoryOrder order;
      if (history_order == "oldest_first") {
        order = ec_native::HistoryOrder::kOldestFirst;
      } else if (history_order == "newest_first") {
        order = ec_native::HistoryOrder::kNewestFirst;
      } else {
        throw std::runtime_error("unsupported observation history order");
      }
      ec_native::ResetFill fill;
      if (reset_fill == "repeat_first") {
        fill = ec_native::ResetFill::kRepeatFirst;
      } else if (reset_fill == "zero") {
        fill = ec_native::ResetFill::kZero;
      } else {
        throw std::runtime_error("unsupported observation reset fill");
      }
      term_configs.push_back(
          {name, width, history_length, history_stride, order, fill});
    }
    core_ = std::make_unique<ec_native::NativeTrackerCore>(
        policy_path, input_name, output_name, term_configs, command_width, defaults,
        scales, lower, upper, half, fsq_z_dim, raw_action_clip,
        intra_op_threads);
  }

  void reset() { core_->reset(); }

  void warmup(std::size_t iterations) {
    py::gil_scoped_release release;
    core_->warmup(iterations);
  }

  py::dict step_once(const FloatArray& joint_position,
                     const FloatArray& joint_velocity,
                     const FloatArray& projected_gravity,
                     const FloatArray& base_angular_velocity,
                     const FloatArray& command) {
    ec_native::RobotState state;
    copy_array(joint_position, state.joint_position, "joint_position");
    copy_array(joint_velocity, state.joint_velocity, "joint_velocity");
    copy_array(projected_gravity, state.projected_gravity,
               "projected_gravity");
    copy_array(base_angular_velocity, state.base_angular_velocity,
               "base_angular_velocity");
    const auto command_info = command.request();
    if (command_info.ndim != 1 ||
        command_info.shape[0] !=
            static_cast<py::ssize_t>(core_->command_width())) {
      throw std::runtime_error("command width mismatch");
    }
    const auto* command_data = static_cast<const float*>(command_info.ptr);
    const ec_native::StepResult* result = nullptr;
    {
      py::gil_scoped_release release;
      result = &core_->step(
          state, std::span<const float>(command_data, core_->command_width()));
    }
    py::array_t<float> observation(
        static_cast<py::ssize_t>(result->observation_width));
    py::array_t<float> action(ec_native::kJointCount);
    py::array_t<float> joint_target(ec_native::kJointCount);
    std::memcpy(observation.mutable_data(), result->observation.data(),
                sizeof(float) * result->observation_width);
    std::memcpy(action.mutable_data(), result->action.data(),
                sizeof(float) * ec_native::kJointCount);
    std::memcpy(joint_target.mutable_data(), result->joint_target.data(),
                sizeof(float) * ec_native::kJointCount);
    py::dict output;
    output["observation"] = std::move(observation);
    output["action"] = std::move(action);
    output["joint_target"] = std::move(joint_target);
    return output;
  }

  std::size_t observation_width() const noexcept {
    return core_->observation_width();
  }

  std::size_t command_width() const noexcept { return core_->command_width(); }

  ec_native::NativeTrackerCore& core() noexcept { return *core_; }

 private:
  std::unique_ptr<ec_native::NativeTrackerCore> core_;
};

class NativeFakeRuntimeBinding {
 public:
  NativeFakeRuntimeBinding(
      NativeTrackerCoreBinding& tracker, const std::string& response_slot,
      const std::string& request_slot, bool create_slots,
      std::size_t control_hz, float lag_alpha,
      std::size_t command_absent_ticks, double command_stale_ms,
      std::size_t hold_steps, std::size_t lead_ticks,
      std::size_t plan_slots, bool latent_plan,
      std::size_t root_qpos_width, std::size_t window_frames,
      std::size_t z_dim, bool sin_cos_phase, std::uint32_t direct_tag,
      bool oracle_reference,
      const std::string& reference_encoder_layout,
      std::size_t encoder_frame_stride,
      const std::string& encoder_trigger_mode,
      const std::string& encoder_path,
      const std::string& encoder_input_name,
      const std::string& encoder_output_name,
      std::size_t encoder_input_width, std::size_t encoder_output_width,
      int cpu, int fifo_priority, bool lock_memory,
      bool require_realtime) {
    ec_native::NativeSchedulerConfig scheduler{
        .control_hz = control_hz,
        .lag_alpha = lag_alpha,
        .command_absent_ticks = command_absent_ticks,
        .command_stale_ms = command_stale_ms,
        .cpu = cpu,
        .fifo_priority = fifo_priority,
        .lock_memory = lock_memory,
        .require_realtime = require_realtime,
    };
    ec_native::NativePlannerConfig planner{
        .hold_steps = hold_steps,
        .lead_ticks = lead_ticks,
        .plan_slots = plan_slots,
        .latent_plan = latent_plan,
        .encoder_frame_width = root_qpos_width,
        .window_frames = window_frames,
        .encoder_frame_stride = encoder_frame_stride,
        .z_dim = z_dim,
        .sin_cos_phase = sin_cos_phase,
        .direct_tag = direct_tag,
        .oracle_reference = oracle_reference,
        .reference_encoder_layout = reference_layout(reference_encoder_layout),
        .encoder_trigger = encoder_trigger(encoder_trigger_mode),
    };
    runtime_ = std::make_unique<ec_native::NativeFakeRuntime>(
        tracker.core(), response_slot, request_slot, create_slots, scheduler,
        planner, encoder_path, encoder_input_name, encoder_output_name,
        encoder_input_width, encoder_output_width);
  }

  void set_anchor_source(const std::string& value) {
    runtime_->set_anchor_source(anchor_source(value));
  }

  void start(std::size_t max_ticks, bool paced) {
    py::gil_scoped_release release;
    runtime_->start(max_ticks, paced);
  }

  void stop() noexcept { runtime_->stop(); }
  void set_reference_paused(bool paused) noexcept {
    runtime_->set_reference_paused(paused);
  }

  void wait() {
    py::gil_scoped_release release;
    runtime_->wait();
  }

  bool running() const noexcept { return runtime_->running(); }

  py::dict stats() const {
    const auto stats = runtime_->stats();
    py::dict result;
    result["ticks"] = stats.ticks;
    result["control_ticks"] = stats.control_ticks;
    result["wait_ticks"] = stats.wait_ticks;
    result["damp_ticks"] = stats.damp_ticks;
    result["deadline_misses"] = stats.deadline_misses;
    result["planner_requests"] = stats.planner_requests;
    result["planner_responses"] = stats.planner_responses;
    result["encoder_inferences"] = stats.encoder_inferences;
    result["response_overruns"] = stats.response_overruns;
    result["stale_responses"] = stats.stale_responses;
    result["reference_ticks"] = stats.reference_ticks;
    result["command_age_ms"] = stats.command_age_ms;
    result["plan_slot_advances"] = stats.plan_slot_advances;
    result["plan_late_starts"] = stats.plan_late_starts;
    result["scheduler_deadlines_missed"] =
        stats.scheduler_deadlines_missed;
    result["tick_ns_max"] = stats.tick_ns_max;
    result["wake_late_ns_max"] = stats.wake_late_ns_max;
    result["last_chunk_offset_steps"] = stats.last_chunk_offset_steps;
    result["backend_steps"] = stats.backend_steps;
    result["backend_wake_late_ns_max"] = stats.backend_wake_late_ns_max;
    result["backend_deadline_misses"] = stats.backend_deadline_misses;
    result["mode"] = static_cast<std::uint32_t>(stats.mode);
    result["fault"] = static_cast<std::uint32_t>(stats.fault);
    result["realtime_configured"] = stats.realtime_configured;
    result["backend_realtime_configured"] =
        stats.backend_realtime_configured;
    return result;
  }

  py::dict state() const {
    const auto state = runtime_->state();
    py::array_t<float> joint_position(ec_native::kJointCount);
    py::array_t<float> joint_velocity(ec_native::kJointCount);
    py::array_t<float> anchor_position_w(3);
    py::array_t<float> anchor_quaternion_w(4);
    std::memcpy(joint_position.mutable_data(), state.joint_position.data(),
                sizeof(float) * ec_native::kJointCount);
    std::memcpy(joint_velocity.mutable_data(), state.joint_velocity.data(),
                sizeof(float) * ec_native::kJointCount);
    std::memcpy(anchor_position_w.mutable_data(), state.anchor_position_w.data(),
                sizeof(float) * 3);
    std::memcpy(anchor_quaternion_w.mutable_data(),
                state.anchor_quaternion_w.data(), sizeof(float) * 4);
    py::dict result;
    result["joint_position"] = std::move(joint_position);
    result["joint_velocity"] = std::move(joint_velocity);
    result["anchor_position_w"] = std::move(anchor_position_w);
    result["anchor_quaternion_w"] = std::move(anchor_quaternion_w);
    result["anchor_pose_valid"] = state.anchor_pose_valid;
    return result;
  }

  py::array_t<std::uint64_t> tick_durations_ns() const {
    const auto durations = runtime_->tick_durations_ns();
    py::array_t<std::uint64_t> result(
        static_cast<py::ssize_t>(durations.size()));
    std::memcpy(result.mutable_data(), durations.data(),
                sizeof(std::uint64_t) * durations.size());
    return result;
  }

  py::array_t<float> base_heights() const {
    const auto values = runtime_->base_heights();
    py::array_t<float> result(static_cast<py::ssize_t>(values.size()));
    std::memcpy(result.mutable_data(), values.data(),
                sizeof(float) * values.size());
    return result;
  }

  py::array_t<float> reference_joint_mae() const {
    const auto values = runtime_->reference_joint_mae();
    py::array_t<float> result(static_cast<py::ssize_t>(values.size()));
    std::memcpy(result.mutable_data(), values.data(),
                sizeof(float) * values.size());
    return result;
  }

  void set_initial_pose(const FloatArray& pose) {
    const auto info = pose.request();
    runtime_->set_initial_pose(std::span<const float>(
        static_cast<const float*>(info.ptr),
        static_cast<std::size_t>(info.size)));
  }

  py::array_t<std::int32_t> reference_frames() const {
    const auto values = runtime_->reference_frames();
    py::array_t<std::int32_t> result(static_cast<py::ssize_t>(values.size()));
    std::memcpy(result.mutable_data(), values.data(),
                sizeof(std::int32_t) * values.size());
    return result;
  }

  py::array_t<float> joint_position_log() const {
    const auto values = runtime_->joint_position_log();
    py::array_t<float> result(static_cast<py::ssize_t>(values.size()));
    std::memcpy(result.mutable_data(), values.data(),
                sizeof(float) * values.size());
    return result;
  }

  py::array_t<float> command_target_log() const {
    const auto values = runtime_->command_target_log();
    py::array_t<float> result(static_cast<py::ssize_t>(values.size()));
    std::memcpy(result.mutable_data(), values.data(),
                sizeof(float) * values.size());
    return result;
  }

  py::array_t<float> anchor_pose_log() const {
    const auto values = runtime_->anchor_pose_log();
    py::array_t<float> result(static_cast<py::ssize_t>(values.size()));
    std::memcpy(result.mutable_data(), values.data(),
                sizeof(float) * values.size());
    return result;
  }

  double backend_time() const { return runtime_->backend_time(); }

  double base_height() const { return runtime_->base_height(); }

  double min_base_height() const { return runtime_->min_base_height(); }

 protected:
  explicit NativeFakeRuntimeBinding(
      std::unique_ptr<ec_native::NativeFakeRuntime> runtime)
      : runtime_(std::move(runtime)) {}

  std::unique_ptr<ec_native::NativeFakeRuntime> runtime_;
};

class NativeMujocoRuntimeBinding : public NativeFakeRuntimeBinding {
 public:
  NativeMujocoRuntimeBinding(
      NativeTrackerCoreBinding& tracker, const std::string& model_path,
      const std::vector<std::string>& isaac_joint_names,
      const FloatArray& default_joint_position, const FloatArray& stiffness,
      const FloatArray& damping, const FloatArray& armature,
      const FloatArray& effort_limit, double timestep,
      std::size_t decimation, const std::string& response_slot,
      const std::string& request_slot, bool create_slots,
      std::size_t control_hz, std::size_t command_absent_ticks,
      double command_stale_ms, std::size_t hold_steps,
      std::size_t lead_ticks, std::size_t plan_slots, bool latent_plan,
      std::size_t root_qpos_width,
      std::size_t window_frames, std::size_t z_dim, bool sin_cos_phase,
      std::uint32_t direct_tag, bool oracle_reference,
      const std::string& reference_encoder_layout,
      std::size_t encoder_frame_stride,
      const std::string& encoder_trigger_mode,
      const std::string& encoder_path,
      const std::string& encoder_input_name,
      const std::string& encoder_output_name,
      std::size_t encoder_input_width, std::size_t encoder_output_width,
      int cpu, int fifo_priority, bool lock_memory,
      bool require_realtime, int physics_cpu, int physics_fifo_priority,
      bool physics_lock_memory, bool physics_require_realtime,
      double noise_joint_pos, double noise_joint_vel,
      double noise_base_ang_vel, double noise_projected_gravity,
      std::uint64_t noise_seed)
      : NativeFakeRuntimeBinding(make_runtime(
            tracker, model_path, isaac_joint_names, default_joint_position,
            stiffness, damping, armature, effort_limit, timestep, decimation,
            response_slot, request_slot, create_slots, control_hz,
            command_absent_ticks, command_stale_ms, hold_steps, lead_ticks,
            plan_slots, latent_plan,
            root_qpos_width, window_frames, z_dim, sin_cos_phase,
            direct_tag, oracle_reference,
            reference_encoder_layout, encoder_frame_stride,
            encoder_trigger_mode,
            encoder_path, encoder_input_name, encoder_output_name,
            encoder_input_width, encoder_output_width, cpu, fifo_priority,
            lock_memory, require_realtime, physics_cpu,
            physics_fifo_priority, physics_lock_memory,
            physics_require_realtime, noise_joint_pos, noise_joint_vel,
            noise_base_ang_vel, noise_projected_gravity, noise_seed)) {}

 private:
  static std::unique_ptr<ec_native::NativeFakeRuntime> make_runtime(
      NativeTrackerCoreBinding& tracker, const std::string& model_path,
      const std::vector<std::string>& isaac_joint_names,
      const FloatArray& default_joint_position, const FloatArray& stiffness,
      const FloatArray& damping, const FloatArray& armature,
      const FloatArray& effort_limit, double timestep,
      std::size_t decimation, const std::string& response_slot,
      const std::string& request_slot, bool create_slots,
      std::size_t control_hz, std::size_t command_absent_ticks,
      double command_stale_ms, std::size_t hold_steps,
      std::size_t lead_ticks, std::size_t plan_slots, bool latent_plan,
      std::size_t root_qpos_width,
      std::size_t window_frames, std::size_t z_dim, bool sin_cos_phase,
      std::uint32_t direct_tag, bool oracle_reference,
      const std::string& reference_encoder_layout,
      std::size_t encoder_frame_stride,
      const std::string& encoder_trigger_mode,
      const std::string& encoder_path,
      const std::string& encoder_input_name,
      const std::string& encoder_output_name,
      std::size_t encoder_input_width, std::size_t encoder_output_width,
      int cpu, int fifo_priority, bool lock_memory,
      bool require_realtime, int physics_cpu, int physics_fifo_priority,
      bool physics_lock_memory, bool physics_require_realtime,
      double noise_joint_pos, double noise_joint_vel,
      double noise_base_ang_vel, double noise_projected_gravity,
      std::uint64_t noise_seed) {
    if (control_hz == 0 ||
        std::abs(timestep * static_cast<double>(decimation) -
                 1.0 / static_cast<double>(control_hz)) >
            1.0e-9) {
      throw std::runtime_error(
          "MuJoCo timestep times decimation must equal the control period");
    }
    const auto defaults = vector_from_array(
        default_joint_position, ec_native::kJointCount,
        "default_joint_position");
    const auto kp = vector_from_array(stiffness, ec_native::kJointCount,
                                      "stiffness");
    const auto kd = vector_from_array(damping, ec_native::kJointCount,
                                      "damping");
    const auto arm = vector_from_array(armature, ec_native::kJointCount,
                                       "armature");
    const auto effort = vector_from_array(
        effort_limit, ec_native::kJointCount, "effort_limit");
    auto backend = std::make_unique<ec_native::NativeMujocoBackend>(
        model_path, isaac_joint_names, defaults, kp, kd, arm, effort,
        timestep, decimation, physics_cpu, physics_fifo_priority,
        physics_lock_memory, physics_require_realtime,
        ec_native::BackendSensorNoise{
            .joint_pos = static_cast<float>(noise_joint_pos),
            .joint_vel = static_cast<float>(noise_joint_vel),
            .base_ang_vel = static_cast<float>(noise_base_ang_vel),
            .projected_gravity = static_cast<float>(noise_projected_gravity),
            .seed = noise_seed,
        });
    ec_native::NativeSchedulerConfig scheduler{
        .control_hz = control_hz,
        .lag_alpha = 1.0F,
        .command_absent_ticks = command_absent_ticks,
        .command_stale_ms = command_stale_ms,
        .cpu = cpu,
        .fifo_priority = fifo_priority,
        .lock_memory = lock_memory,
        .require_realtime = require_realtime,
    };
    ec_native::NativePlannerConfig planner{
        .hold_steps = hold_steps,
        .lead_ticks = lead_ticks,
        .plan_slots = plan_slots,
        .latent_plan = latent_plan,
        .encoder_frame_width = root_qpos_width,
        .window_frames = window_frames,
        .encoder_frame_stride = encoder_frame_stride,
        .z_dim = z_dim,
        .sin_cos_phase = sin_cos_phase,
        .direct_tag = direct_tag,
        .oracle_reference = oracle_reference,
        .reference_encoder_layout = reference_layout(reference_encoder_layout),
        .encoder_trigger = encoder_trigger(encoder_trigger_mode),
    };
    return std::make_unique<ec_native::NativeFakeRuntime>(
        tracker.core(), response_slot, request_slot, create_slots, scheduler,
        planner, std::move(backend), encoder_path, encoder_input_name,
        encoder_output_name, encoder_input_width, encoder_output_width);
  }
};

#ifdef EC_WITH_UNITREE
class NativeUnitreeRuntimeBinding : public NativeFakeRuntimeBinding {
 public:
  NativeUnitreeRuntimeBinding(
      NativeTrackerCoreBinding& tracker,
      const std::string& network_interface,
      const std::vector<std::size_t>& isaac_to_sdk,
      const FloatArray& default_joint_position, const FloatArray& stiffness,
      const FloatArray& damping, const FloatArray& joint_lower,
      const FloatArray& joint_upper, bool writes_enabled,
      const std::string& response_slot, const std::string& request_slot,
      bool create_slots, const py::dict& config)
      : NativeUnitreeRuntimeBinding(make_runtime(
            tracker, network_interface, isaac_to_sdk,
            default_joint_position, stiffness, damping, joint_lower,
            joint_upper, writes_enabled,
            response_slot, request_slot, create_slots, config)) {}

  bool state_ready() const noexcept { return backend_->state_ready(); }

  void begin_initialization(double duration_seconds,
                            const std::vector<float>& target_position,
                            bool hold_current, bool skip_motion_switcher) {
    backend_->begin_initialization(duration_seconds, target_position,
                                   hold_current, skip_motion_switcher);
  }

  void engage_control(std::size_t blend_ticks) {
    backend_->engage_control(blend_ticks);
  }

  void force_damp() noexcept { backend_->force_damp(); }

  std::string vendor_mode() {
    py::gil_scoped_release release;
    return backend_->vendor_mode();
  }

  void open_damp_gate() { backend_->open_damp_gate(); }

  void release_vendor() {
    py::gil_scoped_release release;
    backend_->release_vendor();
  }

  void hold() { backend_->hold(); }

  void set_ramp_guard(float rad, std::uint32_t ticks) {
    backend_->set_ramp_guard(rad, ticks);
  }

  void set_hold_gain_scale(float scale) {
    backend_->set_hold_gain_scale(scale);
  }

  void clear_latched_fault() noexcept { backend_->clear_latched_fault(); }

  void close_gate() { backend_->close_gate(); }

  void restore_vendor(const std::string& name) {
    py::gil_scoped_release release;
    backend_->restore_vendor(name);
  }

  std::vector<float> command_target_error() const {
    const auto values = backend_->command_target_error();
    return std::vector<float>(values.begin(), values.end());
  }

  py::dict latest_state() const {
    const auto state = backend_->latest_state();
    py::dict result;
    result["valid"] = state.valid;
    result["joint_position"] = std::vector<float>(
        state.joint_position.begin(), state.joint_position.end());
    result["joint_velocity"] = std::vector<float>(
        state.joint_velocity.begin(), state.joint_velocity.end());
    result["projected_gravity"] = std::vector<float>(
        state.projected_gravity.begin(), state.projected_gravity.end());
    return result;
  }

  std::uint32_t unitree_mode() const noexcept {
    return static_cast<std::uint32_t>(backend_->mode());
  }

  py::dict writer_stats() const {
    const auto stats = backend_->writer_stats();
    py::dict result;
    result["ticks"] = stats.ticks;
    result["publishes"] = stats.publishes;
    result["publish_failures"] = stats.publish_failures;
    result["crc_errors"] = stats.crc_errors;
    result["hardware_faults"] = stats.hardware_faults;
    result["hardware_fault_latched"] = stats.hardware_fault_latched;
    result["state_fault_reason"] = stats.state_fault_reason;
    result["state_fault_joint"] = stats.state_fault_joint;
    result["state_fault_sdk_joint"] = stats.state_fault_sdk_joint;
    result["state_fault_motorstate"] = stats.state_fault_motorstate;
    result["watchdog_faults"] = stats.watchdog_faults;
    result["wake_late_ns_max"] = stats.wake_late_ns_max;
    result["deadline_misses"] = stats.deadline_misses;
    result["ramp_faults"] = stats.ramp_faults;
    result["state_frames"] = stats.state_frames;
    result["state_gap_ns_max"] = stats.state_gap_ns_max;
    result["ramp_error_max"] = stats.ramp_error_max;
    result["ramp_error_joint"] = stats.ramp_error_joint;
    result["joint_speed_max"] = stats.joint_speed_max;
    result["tracking_error_max"] = stats.tracking_error_max;
    result["command_target_error_max"] = stats.command_target_error_max;
    result["blend_ticks_remaining"] = stats.blend_ticks_remaining;
    result["anchor_yaw_offset_degrees"] = stats.anchor_yaw_offset_degrees;
    result["anchor_heading_captured"] = stats.anchor_heading_captured;
    static const char* const kAnchorPositionSources[] = {
        "fixed_start", "odometry", "leg_kinematics", "auto"};
    result["anchor_position_source"] =
        stats.anchor_position_source < 4
            ? kAnchorPositionSources[stats.anchor_position_source]
            : "unknown";
    result["odometry_frames"] = stats.odometry_frames;
    result["odometry_stale_ticks"] = stats.odometry_stale_ticks;
    result["anchor_displacement_max"] = stats.anchor_displacement_max;
    result["leg_odometry_stance_switches"] = stats.leg_odometry_stance_switches;
    result["mode"] = static_cast<std::uint32_t>(stats.mode);
    result["writes_enabled"] = stats.writes_enabled;
    result["realtime_configured"] = stats.realtime_configured;
    result["gate_open"] = stats.gate_open;
    result["vendor_released"] = stats.vendor_released;
    return result;
  }

  bool wait_for_state(double timeout_seconds) const {
    py::gil_scoped_release release;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration<double>(timeout_seconds);
    while (std::chrono::steady_clock::now() < deadline) {
      if (backend_->state_ready()) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return backend_->state_ready();
  }

  bool wait_for_mode(std::uint32_t expected, double timeout_seconds) const {
    py::gil_scoped_release release;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration<double>(timeout_seconds);
    while (std::chrono::steady_clock::now() < deadline) {
      if (static_cast<std::uint32_t>(backend_->mode()) == expected) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return static_cast<std::uint32_t>(backend_->mode()) == expected;
  }

 private:
  struct BuildResult {
    std::unique_ptr<ec_native::NativeFakeRuntime> runtime;
    ec_native::NativeUnitreeBackend* backend = nullptr;
  };

  explicit NativeUnitreeRuntimeBinding(BuildResult result)
      : NativeFakeRuntimeBinding(std::move(result.runtime)),
        backend_(result.backend) {}

  template <typename Value>
  static Value config_value_or(const py::dict& config, const char* name,
                               Value fallback) {
    if (!config.contains(name)) {
      return fallback;
    }
    return py::cast<Value>(config[name]);
  }

  template <typename Value>
  static Value config_value(const py::dict& config, const char* name) {
    if (!config.contains(name)) {
      throw std::runtime_error(std::string("native Unitree config misses ") +
                               name);
    }
    return py::cast<Value>(config[name]);
  }

  static BuildResult make_runtime(
      NativeTrackerCoreBinding& tracker,
      const std::string& network_interface,
      const std::vector<std::size_t>& isaac_to_sdk,
      const FloatArray& default_joint_position, const FloatArray& stiffness,
      const FloatArray& damping, const FloatArray& joint_lower,
      const FloatArray& joint_upper, bool writes_enabled,
      const std::string& response_slot, const std::string& request_slot,
      bool create_slots, const py::dict& config) {
    const auto defaults = vector_from_array(
        default_joint_position, ec_native::kJointCount,
        "default_joint_position");
    const auto kp = vector_from_array(stiffness, ec_native::kJointCount,
                                      "stiffness");
    const auto kd = vector_from_array(damping, ec_native::kJointCount,
                                      "damping");
    const auto lower = vector_from_array(joint_lower, ec_native::kJointCount,
                                         "joint_lower");
    const auto upper = vector_from_array(joint_upper, ec_native::kJointCount,
                                         "joint_upper");
    auto backend = std::make_unique<ec_native::NativeUnitreeBackend>(
        network_interface, isaac_to_sdk, defaults, kp, kd, lower, upper,
        writes_enabled,
        config_value<int>(config, "writer_cpu"),
        config_value<int>(config, "writer_fifo_priority"),
        config_value<bool>(config, "lock_memory"),
        config_value<bool>(config, "require_realtime"),
        config_value<double>(config, "state_absent_ms"),
        config_value<double>(config, "command_stale_ms"),
        config_value_or<int>(config, "dds_domain", 0));
    if (config.contains("fixed_anchor_position")) {
      backend->set_fixed_anchor_pose(
          py::cast<std::vector<float>>(config["fixed_anchor_position"]),
          py::cast<std::vector<float>>(config["fixed_anchor_quaternion"]));
    }
    if (config.contains("anchor_position_source")) {
      const std::vector<std::string> isaac_joint_names =
          config.contains("isaac_joint_names")
              ? py::cast<std::vector<std::string>>(config["isaac_joint_names"])
              : std::vector<std::string>{};
      backend->configure_live_anchor(
          anchor_position_source(
              config_value<std::string>(config, "anchor_position_source")),
          config_value_or<std::string>(config, "odometry_topic", ""),
          config_value_or<std::string>(config, "odometry_mjcf", ""),
          isaac_joint_names);
    }
    ec_native::NativeUnitreeBackend* backend_pointer = backend.get();
    ec_native::NativeSchedulerConfig scheduler{
        .control_hz = config_value<std::size_t>(config, "control_hz"),
        .lag_alpha = 1.0F,
        .command_absent_ticks =
            config_value<std::size_t>(config, "command_absent_ticks"),
        .command_stale_ms =
            config_value<double>(config, "command_stale_ms"),
        .state_absent_ms =
            config_value<double>(config, "state_absent_ms"),
        .cpu = config_value<int>(config, "cpu"),
        .fifo_priority = config_value<int>(config, "fifo_priority"),
        .lock_memory = config_value<bool>(config, "lock_memory"),
        .require_realtime =
            config_value<bool>(config, "require_realtime"),
    };
    ec_native::NativePlannerConfig planner{
        .hold_steps = config_value<std::size_t>(config, "hold_steps"),
        .lead_ticks = config_value<std::size_t>(config, "lead_ticks"),
        .plan_slots =
            config_value_or<std::size_t>(config, "plan_slots", 1),
        .latent_plan = config_value_or<bool>(config, "latent_plan", false),
        .encoder_frame_width =
            config_value<std::size_t>(config, "root_qpos_width"),
        .window_frames =
            config_value<std::size_t>(config, "window_frames"),
        .encoder_frame_stride =
            config_value<std::size_t>(config, "encoder_frame_stride"),
        .z_dim = config_value<std::size_t>(config, "z_dim"),
        .sin_cos_phase = config_value<bool>(config, "sin_cos_phase"),
        .direct_tag = config_value<std::uint32_t>(config, "direct_tag"),
        .oracle_reference =
            config_value_or<bool>(config, "oracle_reference", false),
        .reference_encoder_layout = reference_layout(
            config_value<std::string>(config, "reference_encoder_layout")),
        .encoder_trigger = encoder_trigger(
            config_value<std::string>(config, "encoder_trigger")),
    };
    auto runtime = std::make_unique<ec_native::NativeFakeRuntime>(
        tracker.core(), response_slot, request_slot, create_slots, scheduler,
        planner, std::move(backend),
        config_value<std::string>(config, "encoder_path"),
        config_value<std::string>(config, "encoder_input_name"),
        config_value<std::string>(config, "encoder_output_name"),
        config_value<std::size_t>(config, "encoder_input_width"),
        config_value<std::size_t>(config, "encoder_output_width"));
    runtime->set_anchor_source(anchor_source(
        config_value_or<std::string>(config, "anchor_source", "robot")));
    return BuildResult{std::move(runtime), backend_pointer};
  }

  ec_native::NativeUnitreeBackend* backend_ = nullptr;
};

class MujocoDdsPlantBinding {
 public:
  MujocoDdsPlantBinding(
      const std::string& model_path, const std::string& network_interface,
      const std::vector<std::string>& sdk_joint_names,
      const FloatArray& default_joint_position, const FloatArray& armature,
      const FloatArray& effort_limit, const FloatArray& hold_stiffness,
      const FloatArray& hold_damping, double timestep, int mode_machine,
      int physics_cpu, int physics_fifo_priority, bool lock_memory,
      bool require_realtime, double noise_joint_pos, double noise_joint_vel,
      double noise_base_ang_vel, double noise_imu_tilt_rad,
      std::uint64_t noise_seed, std::size_t state_log_capacity,
      int dds_domain, bool freeze_until_command, bool vendor_enabled,
      const std::string& vendor_name, bool hoist_enabled,
      double hoist_clearance)
      : plant_(model_path, network_interface, sdk_joint_names,
               vector_from_array(default_joint_position, ec_native::kJointCount,
                                 "default_joint_position"),
               vector_from_array(armature, ec_native::kJointCount, "armature"),
               vector_from_array(effort_limit, ec_native::kJointCount,
                                 "effort_limit"),
               vector_from_array(hold_stiffness, ec_native::kJointCount,
                                 "hold_stiffness"),
               vector_from_array(hold_damping, ec_native::kJointCount,
                                 "hold_damping"),
               timestep, static_cast<std::uint8_t>(mode_machine), physics_cpu,
               physics_fifo_priority, lock_memory, require_realtime,
               ec_native::PlantSensorNoise{
                   .joint_pos = static_cast<float>(noise_joint_pos),
                   .joint_vel = static_cast<float>(noise_joint_vel),
                   .base_ang_vel = static_cast<float>(noise_base_ang_vel),
                   .imu_tilt_rad = static_cast<float>(noise_imu_tilt_rad),
                   .seed = noise_seed,
               },
               state_log_capacity, dds_domain, freeze_until_command,
               vendor_enabled, vendor_name, hoist_enabled, hoist_clearance) {
    if (mode_machine < 0 || mode_machine > 255) {
      throw std::runtime_error("mode_machine must fit in one byte");
    }
  }

  void start() { plant_.start(); }
  void publish_odometry(const std::string& topic) {
    plant_.publish_odometry(topic);
  }
  void stop() { plant_.stop(); }
  void wait_for_stop() {
    py::gil_scoped_release release;
    plant_.wait_for_stop();
  }
  bool running() const { return plant_.running(); }
  void reset() { plant_.reset(); }
  void hoist() { plant_.hoist(); }
  void lower() { plant_.lower(); }
  void slack() { plant_.slack(); }

  void set_initial_pose(const FloatArray& pose) {
    const auto values =
        vector_from_array(pose, 36, "initial_pose", /*allow_empty=*/true);
    plant_.set_initial_pose(values);
  }

  py::array_t<float> latest_state() const {
    const auto values = plant_.latest_state();
    py::array_t<float> out(static_cast<py::ssize_t>(values.size()));
    std::copy(values.begin(), values.end(), out.mutable_data());
    return out;
  }

  py::array_t<float> state_log() const {
    const std::vector<float> values = plant_.state_log();
    const std::size_t rows = values.size() / 36;
    py::array_t<float> out({static_cast<py::ssize_t>(rows),
                            static_cast<py::ssize_t>(36)});
    std::copy(values.begin(), values.end(), out.mutable_data());
    return out;
  }

  py::array_t<float> hoist_log() const {
    const auto values = plant_.hoist_log();
    py::array_t<float> out({static_cast<py::ssize_t>(values.size() / 9), static_cast<py::ssize_t>(9)});
    std::copy(values.begin(), values.end(), out.mutable_data());
    return out;
  }

  py::array_t<double> hoist_attachment_points() const {
    const auto values = plant_.hoist_attachment_points();
    py::array_t<double> out({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(3)});
    std::copy(values.begin(), values.end(), out.mutable_data());
    return out;
  }

  py::dict stats() const {
    const ec_native::PlantStats values = plant_.stats();
    py::dict result;
    result["steps"] = values.steps;
    result["publishes"] = values.publishes;
    result["publish_failures"] = values.publish_failures;
    result["commands_received"] = values.commands_received;
    result["rejected_commands"] = values.rejected_commands;
    result["mode_machine_rejections"] = values.mode_machine_rejections;
    result["crc_errors"] = values.crc_errors;
    result["wake_late_ns_max"] = values.wake_late_ns_max;
    result["deadline_misses"] = values.deadline_misses;
    result["holding"] = values.holding;
    result["vendor_owned"] = values.vendor_owned;
    result["vendor_fsm_id"] = values.vendor_fsm_id;
    result["hoisted"] = values.hoisted;
    result["hoist_mode"] = values.hoist_mode;
    result["hoist_gain"] = values.hoist_gain;
    result["hoist_clearance"] = values.hoist_clearance;
    result["foot_clearance"] = values.foot_clearance;
    result["physics_fault"] = values.physics_fault;
    result["realtime_configured"] = values.realtime_configured;
    result["last_command_age_ms"] = values.last_command_age_ms;
    result["time"] = values.time;
    result["base_height"] = values.base_height;
    result["min_base_height"] = values.min_base_height;
    result["applied_kp_min"] = values.applied_kp_min;
    result["applied_kp_max"] = values.applied_kp_max;
    result["applied_kd_min"] = values.applied_kd_min;
    result["applied_kd_max"] = values.applied_kd_max;
    result["applied_q_absmax"] = values.applied_q_absmax;
    result["applied_extra_absmax"] = values.applied_extra_absmax;
    return result;
  }

 private:
  ec_native::MujocoDdsPlant plant_;
};
#endif

class OnnxEngineBinding {
 public:
  OnnxEngineBinding(const std::string& model_path,
                    const std::string& input_name,
                    const std::string& output_name, std::size_t input_width,
                    std::size_t output_width, std::size_t intra_op_threads)
      : engine_(model_path, input_name, output_name, input_width,
                output_width, intra_op_threads) {}

  void warmup(std::size_t iterations) {
    py::gil_scoped_release release;
    engine_.warmup(iterations);
  }

  py::array_t<float> infer(const FloatArray& input) {
    const auto info = input.request();
    if (info.ndim != 1 ||
        info.shape[0] != static_cast<py::ssize_t>(engine_.input_width())) {
      throw std::runtime_error("ONNX input width mismatch");
    }
    const auto* data = static_cast<const float*>(info.ptr);
    std::span<const float> result;
    {
      py::gil_scoped_release release;
      result = engine_.infer(
          std::span<const float>(data, engine_.input_width()));
    }
    py::array_t<float> output(
        static_cast<py::ssize_t>(engine_.output_width()));
    std::memcpy(output.mutable_data(), result.data(),
                sizeof(float) * result.size());
    return output;
  }

 private:
  ec_native::OnnxEngine engine_;
};

}  // namespace

PYBIND11_MODULE(_ec_native, m) {
  m.doc() = "Native hot-path modules for embodied_control.lowlevel";
  m.def("monotonic_now", &ec_native::monotonic_now,
        "CLOCK_MONOTONIC seconds (same clock as Python time.monotonic on Linux)");
  m.def("reexpress_root_qpos_window", &reexpress_root_qpos_binding,
        py::arg("raw_world_frames"), py::arg("anchor_position_w"),
        py::arg("anchor_quaternion_w"), py::arg("heading_only") = false);
  m.def("projected_gravity_from_xyzw", &projected_gravity_binding,
        py::arg("quaternion_xyzw"));
  m.def("align_heading_to_reference", &align_heading_binding,
        py::arg("initial_robot"), py::arg("initial_reference"),
        py::arg("current_robot"));
  m.def("pack_joint_qpos_qvel_anchor_ori_window",
        &pack_joint_reference_binding, py::arg("raw_world_frames"),
        py::arg("start_frame"), py::arg("frame_count"),
        py::arg("frame_stride"), py::arg("robot_quaternion_w"));
  m.attr("MAX_VALUES") = py::int_(ec_native::kMaxValues);
#ifdef EC_WITH_UNITREE
  m.attr("WITH_UNITREE") = py::bool_(true);
#else
  m.attr("WITH_UNITREE") = py::bool_(false);
#endif

  py::class_<ShmCommandSlot>(m, "ShmCommandSlot")
      .def(py::init<const std::string&, bool>(), py::arg("name"),
           py::arg("create"))
      .def("publish", &ShmCommandSlot::publish, py::arg("sequence"),
           py::arg("interface_tag"), py::arg("values"),
           py::arg("sender_stamp") = 0.0)
      .def("snapshot", &ShmCommandSlot::snapshot,
           py::arg("after_sequence") = 0);

  py::class_<NativeTrackerCoreBinding>(m, "NativeTrackerCore")
      .def(py::init<const std::string&, const std::string&,
                    const std::string&,
                    const std::vector<NativeTrackerCoreBinding::TermTuple>&,
                    std::size_t, const FloatArray&, const FloatArray&,
                    const FloatArray&, const FloatArray&, const FloatArray&,
                    std::size_t, float, std::size_t>(),
           py::arg("policy_path"), py::arg("input_name"),
           py::arg("output_name"), py::arg("terms"),
           py::arg("command_width"), py::arg("default_joint_position"),
           py::arg("action_scale"), py::arg("joint_lower"),
           py::arg("joint_upper"), py::arg("fsq_half_levels"),
           py::arg("fsq_z_dim"), py::arg("raw_action_clip") = 0.0F,
           py::arg("intra_op_threads") = 4)
      .def("reset", &NativeTrackerCoreBinding::reset)
      .def("warmup", &NativeTrackerCoreBinding::warmup,
           py::arg("iterations") = 8)
      .def("step_once", &NativeTrackerCoreBinding::step_once,
           py::arg("joint_position"), py::arg("joint_velocity"),
           py::arg("projected_gravity"),
           py::arg("base_angular_velocity"), py::arg("command"))
      .def_property_readonly("observation_width",
                             &NativeTrackerCoreBinding::observation_width)
      .def_property_readonly("command_width",
                             &NativeTrackerCoreBinding::command_width);

  py::class_<OnnxEngineBinding>(m, "OnnxEngine")
      .def(py::init<const std::string&, const std::string&,
                    const std::string&, std::size_t, std::size_t,
                    std::size_t>(),
           py::arg("model_path"), py::arg("input_name"),
           py::arg("output_name"), py::arg("input_width"),
           py::arg("output_width"), py::arg("intra_op_threads") = 4)
      .def("warmup", &OnnxEngineBinding::warmup, py::arg("iterations") = 8)
      .def("infer", &OnnxEngineBinding::infer, py::arg("input"));

  py::class_<LegOdometryBinding>(m, "LegOdometry")
      .def(py::init<const std::string&, const std::vector<std::string>&,
                    float>(),
           py::arg("mjcf_path"), py::arg("isaac_joint_names"),
           py::arg("contact_threshold_m") = 0.005F)
      .def("reset", &LegOdometryBinding::reset)
      .def("update", &LegOdometryBinding::update, py::arg("joint_position"),
           py::arg("quaternion_xyzw"))
      .def_property_readonly("stance_foot", &LegOdometryBinding::stance_foot)
      .def_property_readonly("stance_switches",
                             &LegOdometryBinding::stance_switches);

  py::class_<NativeFakeRuntimeBinding>(m, "NativeFakeRuntime")
      .def(py::init<
               NativeTrackerCoreBinding&, const std::string&,
               const std::string&, bool, std::size_t, float, std::size_t,
               double, std::size_t, std::size_t, std::size_t, bool,
               std::size_t,
               std::size_t, std::size_t, bool, std::uint32_t, bool,
               const std::string&, std::size_t, const std::string&,
               const std::string&, const std::string&, const std::string&,
               std::size_t, std::size_t, int, int, bool, bool>(),
           py::arg("tracker"), py::arg("response_slot"),
           py::arg("request_slot") = "", py::arg("create_slots") = true,
           py::arg("control_hz") = 50, py::arg("lag_alpha") = 1.0F,
           py::arg("command_absent_ticks") = 100,
           py::arg("command_stale_ms") = 500.0,
           py::arg("hold_steps") = 10, py::arg("lead_ticks") = 4,
           py::arg("plan_slots") = 1, py::arg("latent_plan") = false,
           py::arg("root_qpos_width") = 38,
           py::arg("window_frames") = 10, py::arg("z_dim") = 256,
           py::arg("sin_cos_phase") = true,
           py::arg("direct_tag") = 1,
           py::arg("oracle_reference") = false,
           py::arg("reference_encoder_layout") = "root_qpos",
           py::arg("encoder_frame_stride") = 1,
           py::arg("encoder_trigger") = "on_acceptance",
           py::arg("encoder_path") = "",
           py::arg("encoder_input_name") = "",
           py::arg("encoder_output_name") = "",
           py::arg("encoder_input_width") = 0,
           py::arg("encoder_output_width") = 0, py::arg("cpu") = -1,
           py::arg("fifo_priority") = 0, py::arg("lock_memory") = false,
           py::arg("require_realtime") = false,
           py::keep_alive<1, 2>())
      .def("start", &NativeFakeRuntimeBinding::start,
           py::arg("max_ticks"), py::arg("paced") = true)
      .def("set_anchor_source", &NativeFakeRuntimeBinding::set_anchor_source,
           py::arg("source"))
      .def("stop", &NativeFakeRuntimeBinding::stop)
      .def("set_reference_paused",
           [](NativeFakeRuntimeBinding& self, bool paused) {
             self.set_reference_paused(paused);
           },
           py::arg("paused"))
      .def("wait", &NativeFakeRuntimeBinding::wait)
      .def_property_readonly("running", &NativeFakeRuntimeBinding::running)
      .def("stats", &NativeFakeRuntimeBinding::stats)
      .def("state", &NativeFakeRuntimeBinding::state)
      .def("tick_durations_ns",
           &NativeFakeRuntimeBinding::tick_durations_ns)
      .def("set_initial_pose", &NativeFakeRuntimeBinding::set_initial_pose,
           py::arg("pose"))
      .def("reference_frames", &NativeFakeRuntimeBinding::reference_frames)
      .def("joint_position_log",
           &NativeFakeRuntimeBinding::joint_position_log)
      .def("anchor_pose_log", &NativeFakeRuntimeBinding::anchor_pose_log)
      .def("command_target_log", &NativeFakeRuntimeBinding::command_target_log)
      .def("base_heights", &NativeFakeRuntimeBinding::base_heights)
      .def("reference_joint_mae",
           &NativeFakeRuntimeBinding::reference_joint_mae)
      .def_property_readonly("backend_time",
                             &NativeFakeRuntimeBinding::backend_time)
      .def_property_readonly("base_height",
                             &NativeFakeRuntimeBinding::base_height)
      .def_property_readonly("min_base_height",
                             &NativeFakeRuntimeBinding::min_base_height);

  py::class_<NativeMujocoRuntimeBinding, NativeFakeRuntimeBinding>(
      m, "NativeMujocoRuntime")
      .def(py::init<
               NativeTrackerCoreBinding&, const std::string&,
               const std::vector<std::string>&, const FloatArray&,
               const FloatArray&, const FloatArray&, const FloatArray&,
               const FloatArray&, double, std::size_t, const std::string&,
               const std::string&, bool, std::size_t, std::size_t, double,
               std::size_t, std::size_t, std::size_t, bool, std::size_t,
               std::size_t,
               std::size_t, bool, std::uint32_t, bool, const std::string&,
               std::size_t, const std::string&, const std::string&,
               const std::string&, const std::string&, std::size_t,
               std::size_t, int, int, bool, bool, int, int, bool, bool,
               double, double, double, double, std::uint64_t>(),
           py::arg("tracker"), py::arg("model_path"),
           py::arg("isaac_joint_names"),
           py::arg("default_joint_position"), py::arg("stiffness"),
           py::arg("damping"), py::arg("armature"),
           py::arg("effort_limit"), py::arg("timestep"),
           py::arg("decimation"), py::arg("response_slot"),
           py::arg("request_slot") = "", py::arg("create_slots") = true,
           py::arg("control_hz") = 50,
           py::arg("command_absent_ticks") = 100,
           py::arg("command_stale_ms") = 500.0,
           py::arg("hold_steps") = 10, py::arg("lead_ticks") = 4,
           py::arg("plan_slots") = 1, py::arg("latent_plan") = false,
           py::arg("root_qpos_width") = 38,
           py::arg("window_frames") = 10, py::arg("z_dim") = 256,
           py::arg("sin_cos_phase") = true,
           py::arg("direct_tag") = 1,
           py::arg("oracle_reference") = false,
           py::arg("reference_encoder_layout") = "root_qpos",
           py::arg("encoder_frame_stride") = 1,
           py::arg("encoder_trigger") = "on_acceptance",
           py::arg("encoder_path") = "",
           py::arg("encoder_input_name") = "",
           py::arg("encoder_output_name") = "",
           py::arg("encoder_input_width") = 0,
           py::arg("encoder_output_width") = 0, py::arg("cpu") = -1,
           py::arg("fifo_priority") = 0, py::arg("lock_memory") = false,
           py::arg("require_realtime") = false,
           py::arg("physics_cpu") = -1,
           py::arg("physics_fifo_priority") = 0,
           py::arg("physics_lock_memory") = false,
           py::arg("physics_require_realtime") = false,
           py::arg("noise_joint_pos") = 0.0, py::arg("noise_joint_vel") = 0.0,
           py::arg("noise_base_ang_vel") = 0.0,
           py::arg("noise_projected_gravity") = 0.0,
           py::arg("noise_seed") = 0,
           py::keep_alive<1, 2>());

#ifdef EC_WITH_UNITREE
  py::class_<ec_native::G1LocoClient>(m, "G1LocoClient")
      .def(py::init<const std::string&, int, float>(),
           py::arg("network_interface"), py::arg("dds_domain") = 0,
           py::arg("timeout_seconds") = 5.0F)
      .def_property_readonly("fsm_id",
                             [](const ec_native::G1LocoClient& client) {
                               py::gil_scoped_release release;
                               return client.fsm_id();
                             })
      .def_property_readonly("fsm_mode",
                             [](const ec_native::G1LocoClient& client) {
                               py::gil_scoped_release release;
                               return client.fsm_mode();
                             })
      .def_property_readonly("balance_mode",
                             [](const ec_native::G1LocoClient& client) {
                               py::gil_scoped_release release;
                               return client.balance_mode();
                             })
      .def_property_readonly("stand_height",
                             [](const ec_native::G1LocoClient& client) {
                               py::gil_scoped_release release;
                               return client.stand_height();
                             })
      .def_property_readonly("swing_height",
                             [](const ec_native::G1LocoClient& client) {
                               py::gil_scoped_release release;
                               return client.swing_height();
                             })
      .def("status",
           [](const ec_native::G1LocoClient& client) {
             ec_native::G1LocoStatus s;
             {
               py::gil_scoped_release release;
               s = client.status();
             }
             py::dict out;
             out["fsm_id"] = s.fsm_id;
             out["fsm_mode"] = s.fsm_mode;
             out["balance_mode"] = s.balance_mode;
             out["stand_height"] = s.stand_height;
             out["swing_height"] = s.swing_height;
             return out;
           })
      .def("set_fsm_id", &ec_native::G1LocoClient::set_fsm_id,
           py::arg("fsm_id"), py::call_guard<py::gil_scoped_release>())
      .def("zero_torque", &ec_native::G1LocoClient::zero_torque,
           py::call_guard<py::gil_scoped_release>())
      .def("damp", &ec_native::G1LocoClient::damp,
           py::call_guard<py::gil_scoped_release>())
      .def("squat", &ec_native::G1LocoClient::squat,
           py::call_guard<py::gil_scoped_release>())
      .def("sit", &ec_native::G1LocoClient::sit,
           py::call_guard<py::gil_scoped_release>())
      .def("stand_up", &ec_native::G1LocoClient::stand_up,
           py::call_guard<py::gil_scoped_release>())
      .def("start", &ec_native::G1LocoClient::start,
           py::call_guard<py::gil_scoped_release>())
      .def("balance_stand", &ec_native::G1LocoClient::balance_stand,
           py::call_guard<py::gil_scoped_release>())
      .def("continuous_gait", &ec_native::G1LocoClient::continuous_gait,
           py::arg("enabled"), py::call_guard<py::gil_scoped_release>())
      .def("set_stand_height", &ec_native::G1LocoClient::set_stand_height,
           py::arg("height"), py::call_guard<py::gil_scoped_release>())
      .def("high_stand", &ec_native::G1LocoClient::high_stand,
           py::call_guard<py::gil_scoped_release>())
      .def("low_stand", &ec_native::G1LocoClient::low_stand,
           py::call_guard<py::gil_scoped_release>())
      .def("set_swing_height", &ec_native::G1LocoClient::set_swing_height,
           py::arg("height"), py::call_guard<py::gil_scoped_release>())
      .def("set_speed_mode", &ec_native::G1LocoClient::set_speed_mode,
           py::arg("speed_mode"), py::call_guard<py::gil_scoped_release>())
      .def("move", &ec_native::G1LocoClient::move, py::arg("vx"),
           py::arg("vy"), py::arg("vyaw"), py::arg("continuous") = false,
           py::call_guard<py::gil_scoped_release>())
      .def("stop_move", &ec_native::G1LocoClient::stop_move,
           py::call_guard<py::gil_scoped_release>())
      .def("wave_hand", &ec_native::G1LocoClient::wave_hand,
           py::arg("turn") = false,
           py::call_guard<py::gil_scoped_release>())
      .def("shake_hand", &ec_native::G1LocoClient::shake_hand,
           py::arg("stage") = -1, py::call_guard<py::gil_scoped_release>());

  py::class_<ec_native::OdometryProbe>(m, "OdometryProbe")
      .def(py::init<const std::string&, const std::string&, int>(),
           py::arg("network_interface"), py::arg("topic") = "rt/odommodestate",
           py::arg("dds_domain") = 0)
      .def("wait_for_frames",
           [](const ec_native::OdometryProbe& probe, std::uint64_t count,
              double timeout_seconds) {
             py::gil_scoped_release release;
             return probe.wait_for_frames(count, timeout_seconds);
           },
           py::arg("count"), py::arg("timeout_seconds"))
      .def("snapshot", [](const ec_native::OdometryProbe& probe) {
        const auto s = probe.snapshot();
        py::dict out;
        out["frames"] = s.frames;
        out["nonfinite_frames"] = s.nonfinite_frames;
        out["first_receive_ns"] = s.first_receive_ns;
        out["last_receive_ns"] = s.last_receive_ns;
        out["max_gap_ns"] = s.max_gap_ns;
        out["first_position"] = std::vector<float>(s.first_position.begin(), s.first_position.end());
        out["position"] = std::vector<float>(s.position.begin(), s.position.end());
        out["velocity"] = std::vector<float>(s.velocity.begin(), s.velocity.end());
        out["quaternion_wxyz"] = std::vector<float>(s.quaternion_wxyz.begin(), s.quaternion_wxyz.end());
        out["error_code"] = s.error_code;
        out["mode"] = static_cast<std::uint32_t>(s.mode);
        return out;
      });

  py::class_<ec_native::UnitreeStateProbe>(m, "UnitreeStateProbe")
      .def(py::init<const std::string&, int, std::size_t>(),
           py::arg("network_interface"), py::arg("dds_domain") = 0,
           py::arg("sample_capacity") = 0)
      .def("samples", [](const ec_native::UnitreeStateProbe& probe) {
        // Rows below `captured` are complete: the handler publishes the
        // count with release semantics after writing the row.
        const auto count = static_cast<py::ssize_t>(probe.captured_samples());
        const auto width = static_cast<py::ssize_t>(ec_native::kProbeSampleWidth);
        py::array_t<float> values({count, width});
        std::copy_n(probe.sample_values().data(), count * width, values.mutable_data());
        py::array_t<std::uint64_t> times(count);
        std::copy_n(probe.sample_times_ns().data(), count, times.mutable_data());
        py::array_t<std::uint32_t> ticks(count);
        std::copy_n(probe.sample_ticks().data(), count, ticks.mutable_data());
        py::dict out;
        out["values"] = values;
        out["receive_ns"] = times;
        out["tick"] = ticks;
        out["capacity"] = probe.sample_capacity();
        return out;
      })
      .def("wait_for_samples",
           [](const ec_native::UnitreeStateProbe& probe, std::uint64_t count,
              double timeout) {
             py::gil_scoped_release release;
             return probe.wait_for_samples(count, timeout);
           },
           py::arg("count"), py::arg("timeout_seconds"))
      .def("snapshot", [](const ec_native::UnitreeStateProbe& probe) {
        const auto s = probe.snapshot();
        py::dict out;
        out["samples"] = s.samples;
        out["crc_errors"] = s.crc_errors;
        out["nonfinite_samples"] = s.nonfinite_samples;
        out["motor_error_samples"] = s.motor_error_samples;
        out["duplicate_ticks"] = s.duplicate_ticks;
        out["first_receive_ns"] = s.first_receive_ns;
        out["last_receive_ns"] = s.last_receive_ns;
        out["max_gap_ns"] = s.max_gap_ns;
        out["first_tick"] = s.first_tick;
        out["last_tick"] = s.last_tick;
        out["motor_error_mask"] = s.motor_error_mask;
        out["mode_machine"] = s.mode_machine;
        out["snapshot_consistent"] = s.snapshot_consistent;
        out["joint_position"] = s.joint_position;
        out["joint_velocity"] = s.joint_velocity;
        out["joint_torque"] = s.joint_torque;
        out["motor_state"] = s.motor_state;
        out["first_motor_error_code"] = s.first_motor_error_code;
        out["quaternion"] = s.quaternion;
        out["gyroscope"] = s.gyroscope;
        out["accelerometer"] = s.accelerometer;
        return out;
      });

  py::class_<NativeUnitreeRuntimeBinding, NativeFakeRuntimeBinding>(
      m, "NativeUnitreeRuntime")
      .def(py::init<
               NativeTrackerCoreBinding&, const std::string&,
               const std::vector<std::size_t>&, const FloatArray&,
               const FloatArray&, const FloatArray&, const FloatArray&,
               const FloatArray&, bool,
               const std::string&, const std::string&, bool,
               const py::dict&>(),
           py::arg("tracker"), py::arg("network_interface"),
           py::arg("isaac_to_sdk"), py::arg("default_joint_position"),
           py::arg("stiffness"), py::arg("damping"),
           py::arg("joint_lower"), py::arg("joint_upper"),
           py::arg("writes_enabled"), py::arg("response_slot"),
           py::arg("request_slot"), py::arg("create_slots"),
           py::arg("config"), py::keep_alive<1, 2>())
      .def_property_readonly("state_ready",
                             &NativeUnitreeRuntimeBinding::state_ready)
      .def("wait_for_state", &NativeUnitreeRuntimeBinding::wait_for_state,
           py::arg("timeout_seconds"))
      .def("begin_initialization",
           &NativeUnitreeRuntimeBinding::begin_initialization,
           py::arg("duration_seconds") = 3.0,
           py::arg("target_position") = std::vector<float>{},
           py::arg("hold_current") = false,
           py::arg("skip_motion_switcher") = false)
      .def("wait_for_mode", &NativeUnitreeRuntimeBinding::wait_for_mode,
           py::arg("expected"), py::arg("timeout_seconds"))
      .def("engage_control", &NativeUnitreeRuntimeBinding::engage_control,
           py::arg("blend_ticks") = 0)
      .def("force_damp", &NativeUnitreeRuntimeBinding::force_damp)
      .def("vendor_mode", &NativeUnitreeRuntimeBinding::vendor_mode)
      .def("open_damp_gate", &NativeUnitreeRuntimeBinding::open_damp_gate)
      .def("release_vendor", &NativeUnitreeRuntimeBinding::release_vendor)
      .def("hold", &NativeUnitreeRuntimeBinding::hold)
      .def("set_ramp_guard", &NativeUnitreeRuntimeBinding::set_ramp_guard,
           py::arg("rad"), py::arg("ticks"))
      .def("set_hold_gain_scale",
           &NativeUnitreeRuntimeBinding::set_hold_gain_scale, py::arg("scale"))
      .def("clear_latched_fault",
           &NativeUnitreeRuntimeBinding::clear_latched_fault)
      .def("close_gate", &NativeUnitreeRuntimeBinding::close_gate)
      .def("restore_vendor", &NativeUnitreeRuntimeBinding::restore_vendor,
           py::arg("name"))
      .def("latest_state", &NativeUnitreeRuntimeBinding::latest_state)
      .def("command_target_error",
           &NativeUnitreeRuntimeBinding::command_target_error)
      .def_property_readonly("unitree_mode",
                             &NativeUnitreeRuntimeBinding::unitree_mode)
      .def("writer_stats", &NativeUnitreeRuntimeBinding::writer_stats);

  py::class_<MujocoDdsPlantBinding>(m, "MujocoDdsPlant")
      .def(py::init<const std::string&, const std::string&,
                    const std::vector<std::string>&, const FloatArray&,
                    const FloatArray&, const FloatArray&, const FloatArray&,
                    const FloatArray&, double, int, int, int, bool, bool,
                    double, double, double, double, std::uint64_t,
                    std::size_t, int, bool, bool, const std::string&, bool,
                    double>(),
           py::arg("model_path"), py::arg("network_interface"),
           py::arg("sdk_joint_names"), py::arg("default_joint_position"),
           py::arg("armature"), py::arg("effort_limit"),
           py::arg("hold_stiffness"), py::arg("hold_damping"),
           py::arg("timestep") = 0.002, py::arg("mode_machine") = 5,
           py::arg("physics_cpu") = -1, py::arg("physics_fifo_priority") = 0,
           py::arg("lock_memory") = false,
           py::arg("require_realtime") = false,
           py::arg("noise_joint_pos") = 0.0, py::arg("noise_joint_vel") = 0.0,
           py::arg("noise_base_ang_vel") = 0.0,
           py::arg("noise_imu_tilt_rad") = 0.0, py::arg("noise_seed") = 0,
           py::arg("state_log_capacity") = 0, py::arg("dds_domain") = 0,
           py::arg("freeze_until_command") = false,
           py::arg("vendor_enabled") = false, py::arg("vendor_name") = "ai",
           py::arg("hoist_enabled") = false,
           py::arg("hoist_clearance") = ec_native::kDefaultHoistClearanceMeters)
      .def("start", &MujocoDdsPlantBinding::start)
      .def("publish_odometry", &MujocoDdsPlantBinding::publish_odometry,
           py::arg("topic"))
      .def("hoist", &MujocoDdsPlantBinding::hoist)
      .def("lower", &MujocoDdsPlantBinding::lower)
      .def("slack", &MujocoDdsPlantBinding::slack)
      .def("stop", &MujocoDdsPlantBinding::stop)
      .def("wait_for_stop", &MujocoDdsPlantBinding::wait_for_stop)
      .def_property_readonly("running", &MujocoDdsPlantBinding::running)
      .def("reset", &MujocoDdsPlantBinding::reset)
      .def("set_initial_pose", &MujocoDdsPlantBinding::set_initial_pose,
           py::arg("pose"))
      .def("latest_state", &MujocoDdsPlantBinding::latest_state)
      .def("state_log", &MujocoDdsPlantBinding::state_log)
      .def("hoist_log", &MujocoDdsPlantBinding::hoist_log)
      .def("hoist_attachment_points", &MujocoDdsPlantBinding::hoist_attachment_points)
      .def("stats", &MujocoDdsPlantBinding::stats);

  py::class_<ec_native::PlantClient>(m, "PlantClient")
      .def(py::init<const std::string&, int, float>(),
           py::arg("network_interface"), py::arg("dds_domain") = 0,
           py::arg("timeout_seconds") = 2.0F)
      .def("hoist", &ec_native::PlantClient::hoist,
           py::call_guard<py::gil_scoped_release>())
      .def("lower", &ec_native::PlantClient::lower,
           py::call_guard<py::gil_scoped_release>())
      .def("slack", &ec_native::PlantClient::slack,
           py::call_guard<py::gil_scoped_release>())
      .def("reset", &ec_native::PlantClient::reset,
           py::call_guard<py::gil_scoped_release>())
      .def("status", &ec_native::PlantClient::status,
           py::call_guard<py::gil_scoped_release>());
#endif

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
