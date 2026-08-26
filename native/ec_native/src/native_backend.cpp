#include "native_backend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>

#include <mujoco/mujoco.h>

namespace ec_native {
namespace {

constexpr int kSnapshotAttempts = 8;

std::uint64_t monotonic_ns_mujoco() noexcept {
  timespec now{};
  clock_gettime(CLOCK_MONOTONIC, &now);
  return static_cast<std::uint64_t>(now.tv_sec) * 1000000000ULL +
         static_cast<std::uint64_t>(now.tv_nsec);
}

timespec mujoco_timespec(std::uint64_t nanoseconds) noexcept {
  return timespec{
      static_cast<time_t>(nanoseconds / 1000000000ULL),
      static_cast<long>(nanoseconds % 1000000000ULL),
  };
}

void update_mujoco_max(std::atomic<std::uint64_t>& destination,
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

bool projected_gravity_from_xyzw(std::span<const float> quaternion,
                                 std::span<float> gravity) noexcept {
  if (quaternion.size() != 4 || gravity.size() != 3 ||
      !finite_values(quaternion)) {
    return false;
  }
  const double norm_squared =
      static_cast<double>(quaternion[0]) * quaternion[0] +
      static_cast<double>(quaternion[1]) * quaternion[1] +
      static_cast<double>(quaternion[2]) * quaternion[2] +
      static_cast<double>(quaternion[3]) * quaternion[3];
  if (norm_squared <= 1.0e-12) {
    return false;
  }
  const double inverse_norm = 1.0 / std::sqrt(norm_squared);
  const double x = quaternion[0] * inverse_norm;
  const double y = quaternion[1] * inverse_norm;
  const double z = quaternion[2] * inverse_norm;
  const double w = quaternion[3] * inverse_norm;
  gravity[0] = static_cast<float>(2.0 * (w * y - x * z));
  gravity[1] = static_cast<float>(-2.0 * (y * z + w * x));
  gravity[2] = static_cast<float>(-(1.0 - 2.0 * (x * x + y * y)));
  return true;
}

NativeFakeBackend::NativeFakeBackend(
    std::span<const float> default_joint_position, std::size_t control_hz,
    float lag_alpha)
    : control_hz_(control_hz), lag_alpha_(lag_alpha) {
  if (default_joint_position.size() != kJointCount ||
      !finite_values(default_joint_position) || control_hz_ == 0 ||
      lag_alpha_ <= 0.0F || lag_alpha_ > 1.0F) {
    throw std::runtime_error("invalid native fake backend configuration");
  }
  std::copy(default_joint_position.begin(), default_joint_position.end(),
            default_joint_position_.begin());
  reset();
}

void NativeFakeBackend::reset() {
  state_.joint_position = default_joint_position_;
  state_.joint_velocity.fill(0.0F);
  state_.projected_gravity = {0.0F, 0.0F, -1.0F};
  state_.base_angular_velocity.fill(0.0F);
  state_.anchor_position_w = {0.0F, 0.0F, 0.75F};
  state_.anchor_quaternion_w = {0.0F, 0.0F, 0.0F, 1.0F};
  state_.anchor_pose_valid = true;
}

void NativeFakeBackend::set_initial_pose(std::span<const float> pose) {
  if (pose.size() != 7 + kJointCount || !finite_values(pose)) {
    throw std::runtime_error("initial pose must have 36 finite values");
  }
  std::copy_n(pose.begin(), 3, state_.anchor_position_w.begin());
  std::copy_n(pose.begin() + 3, 4, state_.anchor_quaternion_w.begin());
  std::copy_n(pose.begin() + 7, kJointCount, state_.joint_position.begin());
  state_.joint_velocity.fill(0.0F);
  state_.base_angular_velocity.fill(0.0F);
  state_.anchor_pose_valid = projected_gravity_from_xyzw(
      state_.anchor_quaternion_w, state_.projected_gravity);
}

void NativeFakeBackend::write_target(std::span<const float> target) noexcept {
  const float dt = 1.0F / static_cast<float>(control_hz_);
  for (std::size_t index = 0; index < kJointCount; ++index) {
    const float previous = state_.joint_position[index];
    const float next =
        previous + lag_alpha_ * (target[index] - previous);
    state_.joint_position[index] = next;
    state_.joint_velocity[index] = (next - previous) / dt;
  }
}

void NativeFakeBackend::damp() noexcept {
  state_.joint_velocity.fill(0.0F);
}

struct NativeMujocoBackend::Impl {
  mjModel* model = nullptr;
  mjData* data = nullptr;

  ~Impl() {
    if (data != nullptr) {
      mj_deleteData(data);
    }
    if (model != nullptr) {
      mj_deleteModel(model);
    }
  }
};

struct NativeMujocoBackend::StateSlot {
  std::atomic<std::uint64_t> sequence{0};
  std::array<std::atomic<float>, kJointCount> joint_position{};
  std::array<std::atomic<float>, kJointCount> joint_velocity{};
  std::array<std::atomic<float>, 3> projected_gravity{};
  std::array<std::atomic<float>, 3> base_angular_velocity{};
  std::array<std::atomic<float>, 3> anchor_position_w{};
  std::array<std::atomic<float>, 4> anchor_quaternion_w{};
  std::atomic<bool> anchor_pose_valid{false};
  std::atomic<std::uint64_t> receive_ns{0};
  std::atomic<bool> valid{false};
};

struct NativeMujocoBackend::StateSnapshot {
  RobotState state{};
  std::uint64_t receive_ns = 0;
  bool valid = false;
};

struct NativeMujocoBackend::CommandSlot {
  std::atomic<std::uint64_t> sequence{0};
  std::array<std::atomic<float>, kJointCount> joint_target{};
};

struct NativeMujocoBackend::CommandSnapshot {
  std::array<float, kJointCount> joint_target{};
};

NativeMujocoBackend::NativeMujocoBackend(
    const std::string& model_path,
    const std::vector<std::string>& isaac_joint_names,
    std::span<const float> default_joint_position,
    std::span<const float> stiffness, std::span<const float> damping,
    std::span<const float> armature, std::span<const float> effort_limit,
    double timestep, std::size_t decimation, int physics_cpu,
    int physics_fifo_priority, bool lock_memory, bool require_realtime,
    const BackendSensorNoise& sensor_noise)
    : impl_(std::make_unique<Impl>()),
      state_slot_(std::make_unique<StateSlot>()),
      command_slot_(std::make_unique<CommandSlot>()),
      timestep_(timestep),
      decimation_(decimation),
      sensor_noise_(sensor_noise),
      physics_cpu_(physics_cpu),
      physics_fifo_priority_(physics_fifo_priority),
      lock_memory_(lock_memory),
      require_realtime_(require_realtime) {
  if (isaac_joint_names.size() != kJointCount ||
      default_joint_position.size() != kJointCount ||
      stiffness.size() != kJointCount || damping.size() != kJointCount ||
      armature.size() != kJointCount || effort_limit.size() != kJointCount ||
      !finite_values(default_joint_position) || !finite_values(stiffness) ||
      !finite_values(damping) || !finite_values(armature) ||
      !finite_values(effort_limit) ||
      std::any_of(stiffness.begin(), stiffness.end(),
                  [](float value) { return value < 0.0F; }) ||
      std::any_of(damping.begin(), damping.end(),
                  [](float value) { return value < 0.0F; }) ||
      std::any_of(armature.begin(), armature.end(),
                  [](float value) { return value < 0.0F; }) ||
      std::any_of(effort_limit.begin(), effort_limit.end(),
                  [](float value) { return value <= 0.0F; }) ||
      !std::isfinite(timestep_) || timestep_ <= 0.0 || decimation_ == 0 ||
      physics_cpu_ < -1 || physics_cpu_ >= CPU_SETSIZE ||
      physics_fifo_priority_ < 0 ||
      physics_fifo_priority_ > sched_get_priority_max(SCHED_FIFO)) {
    throw std::runtime_error("invalid native MuJoCo backend configuration");
  }
  std::copy(default_joint_position.begin(), default_joint_position.end(),
            default_joint_position_.begin());
  std::copy(stiffness.begin(), stiffness.end(), stiffness_.begin());
  std::copy(damping.begin(), damping.end(), damping_.begin());

  std::array<char, 1024> error{};
  impl_->model =
      mj_loadXML(model_path.c_str(), nullptr, error.data(), error.size());
  if (impl_->model == nullptr) {
    throw std::runtime_error("MuJoCo model load failed: " +
                             std::string(error.data()));
  }
  impl_->data = mj_makeData(impl_->model);
  if (impl_->data == nullptr) {
    throw std::runtime_error("MuJoCo data allocation failed");
  }
  mjModel* model = impl_->model;
  pelvis_body_id_ = mj_name2id(model, mjOBJ_BODY, "pelvis");
  if (pelvis_body_id_ < 0) {
    throw std::runtime_error("MuJoCo model has no pelvis body");
  }
  model->opt.timestep = timestep_;
  model->opt.integrator = mjINT_IMPLICITFAST;
  if (model->nu != static_cast<int>(kJointCount)) {
    throw std::runtime_error("MuJoCo model must have 29 actuators");
  }

  for (std::size_t actuator = 0; actuator < kJointCount; ++actuator) {
    const int joint = model->actuator_trnid[2 * actuator];
    const char* joint_name = mj_id2name(model, mjOBJ_JOINT, joint);
    if (joint_name == nullptr) {
      throw std::runtime_error("MuJoCo actuator joint has no name");
    }
    const auto found =
        std::find(isaac_joint_names.begin(), isaac_joint_names.end(),
                  std::string(joint_name));
    if (found == isaac_joint_names.end()) {
      throw std::runtime_error("MuJoCo joint is absent from action contract: " +
                               std::string(joint_name));
    }
    const std::size_t isaac =
        static_cast<std::size_t>(found - isaac_joint_names.begin());
    actuator_to_isaac_[actuator] = isaac;
    qpos_address_[actuator] = model->jnt_qposadr[joint];
    const int dof = model->jnt_dofadr[joint];
    dof_address_[actuator] = dof;
    model->dof_armature[dof] = armature[isaac];
    model->dof_damping[dof] = 0.0;
    model->dof_frictionloss[dof] = 0.0;
    model->actuator_forcelimited[actuator] = 1;
    model->actuator_forcerange[2 * actuator] = -effort_limit[isaac];
    model->actuator_forcerange[2 * actuator + 1] = effort_limit[isaac];
    model->actuator_ctrllimited[actuator] = 0;
  }
  set_gains(stiffness_, damping_);
  reset();
}

NativeMujocoBackend::~NativeMujocoBackend() {
  stop();
  wait_for_stop();
}

void NativeMujocoBackend::set_gains(
    std::span<const float> stiffness,
    std::span<const float> damping) noexcept {
  mjModel* model = impl_->model;
  for (std::size_t actuator = 0; actuator < kJointCount; ++actuator) {
    const std::size_t isaac = actuator_to_isaac_[actuator];
    model->actuator_gaintype[actuator] = mjGAIN_FIXED;
    model->actuator_biastype[actuator] = mjBIAS_AFFINE;
    mju_zero(model->actuator_gainprm + actuator * mjNGAIN, mjNGAIN);
    mju_zero(model->actuator_biasprm + actuator * mjNBIAS, mjNBIAS);
    model->actuator_gainprm[actuator * mjNGAIN] = stiffness[isaac];
    model->actuator_biasprm[actuator * mjNBIAS + 1] = -stiffness[isaac];
    model->actuator_biasprm[actuator * mjNBIAS + 2] = -damping[isaac];
  }
}

void NativeMujocoBackend::set_initial_pose(std::span<const float> pose) {
  if (pose.empty()) {
    has_initial_pose_ = false;
    return;
  }
  if (pose.size() != initial_pose_.size()) {
    throw std::runtime_error("initial pose must have 36 values");
  }
  std::copy(pose.begin(), pose.end(), initial_pose_.begin());
  has_initial_pose_ = true;
}

void NativeMujocoBackend::reset() {
  stop();
  wait_for_stop();
  // Seeded per run, so a repeated episode sees the identical noise stream.
  noise_state_ =
      sensor_noise_.seed * 0x9e3779b97f4a7c15ull + 0x123456789abcdefull;
  mj_resetData(impl_->model, impl_->data);
  mjData* data = impl_->data;
  if (impl_->model->nq < 7 || impl_->model->nv < 6) {
    throw std::runtime_error("MuJoCo model has no floating base");
  }
  if (has_initial_pose_) {
    data->qpos[0] = initial_pose_[0];
    data->qpos[1] = initial_pose_[1];
    data->qpos[2] = initial_pose_[2];
    // Runtime quaternions are XYZW; MuJoCo stores WXYZ.
    data->qpos[3] = initial_pose_[6];
    data->qpos[4] = initial_pose_[3];
    data->qpos[5] = initial_pose_[4];
    data->qpos[6] = initial_pose_[5];
  } else {
    data->qpos[0] = 0.0;
    data->qpos[1] = 0.0;
    data->qpos[2] = 0.76;
    data->qpos[3] = 1.0;
    data->qpos[4] = 0.0;
    data->qpos[5] = 0.0;
    data->qpos[6] = 0.0;
  }
  for (std::size_t actuator = 0; actuator < kJointCount; ++actuator) {
    const std::size_t isaac = actuator_to_isaac_[actuator];
    const float joint = has_initial_pose_
                            ? initial_pose_[7 + isaac]
                            : default_joint_position_[isaac];
    data->qpos[qpos_address_[actuator]] = joint;
    data->ctrl[actuator] = joint;
  }
  set_gains(stiffness_, damping_);
  mj_forward(impl_->model, data);
  data->time = 0.0;
  for (std::size_t index = 0; index < kJointCount; ++index) {
    command_slot_->joint_target[index].store(default_joint_position_[index],
                                              std::memory_order_relaxed);
  }
  command_slot_->sequence.store(0, std::memory_order_relaxed);
  state_slot_->sequence.store(0, std::memory_order_relaxed);
  state_slot_->valid.store(false, std::memory_order_relaxed);
  stop_requested_.store(false, std::memory_order_relaxed);
  thread_ready_.store(false, std::memory_order_relaxed);
  damp_requested_.store(false, std::memory_order_relaxed);
  realtime_configured_.store(false, std::memory_order_relaxed);
  physics_fault_.store(false, std::memory_order_relaxed);
  physics_steps_.store(0, std::memory_order_relaxed);
  wake_late_ns_max_.store(0, std::memory_order_relaxed);
  deadline_misses_.store(0, std::memory_order_relaxed);
  last_state_read_ns_.store(0, std::memory_order_relaxed);
  simulation_time_.store(0.0, std::memory_order_relaxed);
  base_height_.store(static_cast<float>(data->qpos[2]),
                     std::memory_order_relaxed);
  min_base_height_.store(static_cast<float>(data->qpos[2]),
                         std::memory_order_relaxed);
  publish_state();
  read_state();
}

const RobotState& NativeMujocoBackend::read_state() noexcept {
  StateSnapshot snapshot;
  if (snapshot_state(snapshot)) {
    state_cache_ = snapshot.state;
    last_state_read_ns_.store(snapshot.receive_ns, std::memory_order_relaxed);
  }
  return state_cache_;
}

void NativeMujocoBackend::write_target(
    std::span<const float> target) noexcept {
  if (target.size() != kJointCount || !finite_values(target)) {
    damp();
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
  std::atomic_thread_fence(std::memory_order_release);
  command_slot_->sequence.store(sequence + 2, std::memory_order_relaxed);
  damp_requested_.store(false, std::memory_order_release);
}

void NativeMujocoBackend::damp() noexcept {
  damp_requested_.store(true, std::memory_order_release);
}

double NativeMujocoBackend::simulation_time() const noexcept {
  return simulation_time_.load(std::memory_order_relaxed);
}

double NativeMujocoBackend::base_height() const noexcept {
  return static_cast<double>(base_height_.load(std::memory_order_relaxed));
}

double NativeMujocoBackend::min_base_height() const noexcept {
  return static_cast<double>(
      min_base_height_.load(std::memory_order_relaxed));
}

void NativeMujocoBackend::start(bool paced) {
  if (!paced) {
    throw std::runtime_error(
        "asynchronous MuJoCo requires paced wall-clock execution");
  }
  if (running_.load(std::memory_order_acquire) || physics_thread_.joinable()) {
    throw std::runtime_error("MuJoCo physics thread is already started");
  }
  stop_requested_.store(false, std::memory_order_relaxed);
  thread_ready_.store(false, std::memory_order_relaxed);
  running_.store(true, std::memory_order_release);
  physics_thread_ = std::thread(&NativeMujocoBackend::physics_loop, this);
  while (!thread_ready_.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  if (require_realtime_ && !realtime_configured_.load(std::memory_order_acquire)) {
    stop();
    wait_for_stop();
    throw std::runtime_error(
        "MuJoCo physics thread did not obtain required real-time settings");
  }
}

void NativeMujocoBackend::stop() noexcept {
  stop_requested_.store(true, std::memory_order_release);
}

void NativeMujocoBackend::wait_for_stop() noexcept {
  if (physics_thread_.joinable()) {
    physics_thread_.join();
  }
}

bool NativeMujocoBackend::configure_physics_thread() noexcept {
  bool configured = true;
  if (lock_memory_ && mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    configured = false;
  }
  if (physics_cpu_ >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(physics_cpu_, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
      configured = false;
    }
  }
  if (physics_fifo_priority_ > 0) {
    sched_param parameters{};
    parameters.sched_priority = physics_fifo_priority_;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &parameters) != 0) {
      configured = false;
    }
  }
  realtime_configured_.store(configured, std::memory_order_release);
  return configured;
}

bool NativeMujocoBackend::snapshot_state(StateSnapshot& destination) const
    noexcept {
  for (int attempt = 0; attempt < kSnapshotAttempts; ++attempt) {
    const std::uint64_t first =
        state_slot_->sequence.load(std::memory_order_acquire);
    if (first == 0 || (first & 1ULL) != 0) {
      continue;
    }
    for (std::size_t index = 0; index < kJointCount; ++index) {
      destination.state.joint_position[index] =
          state_slot_->joint_position[index].load(std::memory_order_relaxed);
      destination.state.joint_velocity[index] =
          state_slot_->joint_velocity[index].load(std::memory_order_relaxed);
    }
    for (std::size_t index = 0; index < 3; ++index) {
      destination.state.projected_gravity[index] =
          state_slot_->projected_gravity[index].load(std::memory_order_relaxed);
      destination.state.base_angular_velocity[index] =
          state_slot_->base_angular_velocity[index].load(
              std::memory_order_relaxed);
    }
    for (std::size_t index = 0; index < 3; ++index) {
      destination.state.anchor_position_w[index] =
          state_slot_->anchor_position_w[index].load(std::memory_order_relaxed);
    }
    for (std::size_t index = 0; index < 4; ++index) {
      destination.state.anchor_quaternion_w[index] =
          state_slot_->anchor_quaternion_w[index].load(
              std::memory_order_relaxed);
    }
    destination.state.anchor_pose_valid =
        state_slot_->anchor_pose_valid.load(std::memory_order_relaxed);
    destination.receive_ns =
        state_slot_->receive_ns.load(std::memory_order_relaxed);
    destination.valid = state_slot_->valid.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (first == state_slot_->sequence.load(std::memory_order_relaxed)) {
      return destination.valid;
    }
  }
  return false;
}

bool NativeMujocoBackend::snapshot_command(CommandSnapshot& destination) const
    noexcept {
  for (int attempt = 0; attempt < kSnapshotAttempts; ++attempt) {
    const std::uint64_t first =
        command_slot_->sequence.load(std::memory_order_acquire);
    if ((first & 1ULL) != 0) {
      continue;
    }
    for (std::size_t index = 0; index < kJointCount; ++index) {
      destination.joint_target[index] =
          command_slot_->joint_target[index].load(std::memory_order_relaxed);
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    if (first == command_slot_->sequence.load(std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

float NativeMujocoBackend::noise_uniform(float half_range) noexcept {
  if (half_range <= 0.0F) {
    return 0.0F;
  }
  noise_state_ += 0x9e3779b97f4a7c15ull;
  std::uint64_t z = noise_state_;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  z ^= z >> 31;
  const float unit =
      static_cast<float>(static_cast<double>(z >> 11) / 9007199254740992.0);
  return (unit * 2.0F - 1.0F) * half_range;
}

void NativeMujocoBackend::publish_state() noexcept {
  const mjData* data = impl_->data;
  RobotState state;
  bool valid = true;
  for (std::size_t actuator = 0; actuator < kJointCount; ++actuator) {
    const std::size_t isaac = actuator_to_isaac_[actuator];
    state.joint_position[isaac] =
        static_cast<float>(data->qpos[qpos_address_[actuator]]);
    state.joint_velocity[isaac] =
        static_cast<float>(data->qvel[dof_address_[actuator]]);
    valid = valid && std::isfinite(state.joint_position[isaac]) &&
            std::isfinite(state.joint_velocity[isaac]);
  }
  state.base_angular_velocity = {
      static_cast<float>(data->qvel[3]),
      static_cast<float>(data->qvel[4]),
      static_cast<float>(data->qvel[5]),
  };
  const std::array<float, 4> root_quaternion_xyzw = {
      static_cast<float>(data->qpos[4]),
      static_cast<float>(data->qpos[5]),
      static_cast<float>(data->qpos[6]),
      static_cast<float>(data->qpos[3]),
  };
  valid = valid && projected_gravity_from_xyzw(
                       root_quaternion_xyzw, state.projected_gravity);
  if (sensor_noise_.active()) {
    // The controller's view only: anchor pose and the metric logs downstream
    // keep the clean values, exactly like the Python rehearsal loop. Gravity
    // is perturbed WITHOUT renormalising, matching Isaac's convention.
    for (std::size_t isaac = 0; isaac < kJointCount; ++isaac) {
      state.joint_position[isaac] += noise_uniform(sensor_noise_.joint_pos);
      state.joint_velocity[isaac] += noise_uniform(sensor_noise_.joint_vel);
    }
    for (std::size_t index = 0; index < 3; ++index) {
      state.base_angular_velocity[index] +=
          noise_uniform(sensor_noise_.base_ang_vel);
      state.projected_gravity[index] +=
          noise_uniform(sensor_noise_.projected_gravity);
    }
  }
  state.anchor_position_w = {
      static_cast<float>(data->xpos[3 * pelvis_body_id_]),
      static_cast<float>(data->xpos[3 * pelvis_body_id_ + 1]),
      static_cast<float>(data->xpos[3 * pelvis_body_id_ + 2]),
  };
  state.anchor_quaternion_w = {
      static_cast<float>(data->xquat[4 * pelvis_body_id_ + 1]),
      static_cast<float>(data->xquat[4 * pelvis_body_id_ + 2]),
      static_cast<float>(data->xquat[4 * pelvis_body_id_ + 3]),
      static_cast<float>(data->xquat[4 * pelvis_body_id_]),
  };
  state.anchor_pose_valid = finite_values(state.anchor_position_w) &&
                            finite_values(state.anchor_quaternion_w);
  valid = valid && finite_values(state.base_angular_velocity) &&
          finite_values(state.projected_gravity) &&
          state.anchor_pose_valid &&
          std::isfinite(data->time) && std::isfinite(data->qpos[2]);

  const std::uint64_t sequence =
      state_slot_->sequence.load(std::memory_order_relaxed);
  state_slot_->sequence.store(sequence + 1, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  for (std::size_t index = 0; index < kJointCount; ++index) {
    state_slot_->joint_position[index].store(state.joint_position[index],
                                              std::memory_order_relaxed);
    state_slot_->joint_velocity[index].store(state.joint_velocity[index],
                                              std::memory_order_relaxed);
  }
  for (std::size_t index = 0; index < 3; ++index) {
    state_slot_->projected_gravity[index].store(
        state.projected_gravity[index], std::memory_order_relaxed);
    state_slot_->base_angular_velocity[index].store(
        state.base_angular_velocity[index], std::memory_order_relaxed);
  }
  for (std::size_t index = 0; index < 3; ++index) {
    state_slot_->anchor_position_w[index].store(
        state.anchor_position_w[index], std::memory_order_relaxed);
  }
  for (std::size_t index = 0; index < 4; ++index) {
    state_slot_->anchor_quaternion_w[index].store(
        state.anchor_quaternion_w[index], std::memory_order_relaxed);
  }
  state_slot_->anchor_pose_valid.store(state.anchor_pose_valid,
                                        std::memory_order_relaxed);
  state_slot_->receive_ns.store(monotonic_ns_mujoco(),
                                std::memory_order_relaxed);
  state_slot_->valid.store(valid, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  state_slot_->sequence.store(sequence + 2, std::memory_order_relaxed);
  simulation_time_.store(data->time, std::memory_order_relaxed);
  base_height_.store(static_cast<float>(data->qpos[2]),
                     std::memory_order_relaxed);
  const float height = static_cast<float>(data->qpos[2]);
  float previous_min = min_base_height_.load(std::memory_order_relaxed);
  while (height < previous_min &&
         !min_base_height_.compare_exchange_weak(
             previous_min, height, std::memory_order_relaxed)) {
  }
  if (!valid) {
    physics_fault_.store(true, std::memory_order_release);
  }
}

void NativeMujocoBackend::physics_loop() noexcept {
  const bool configured = configure_physics_thread();
  thread_ready_.store(true, std::memory_order_release);
  if (!configured && require_realtime_) {
    physics_fault_.store(true, std::memory_order_release);
    running_.store(false, std::memory_order_release);
    return;
  }

  const auto period_ns = static_cast<std::uint64_t>(
      std::llround(timestep_ * 1000000000.0));
  // Run physics halfway between control boundaries. The controller writes at
  // t = 0, 20, ... ms; physics samples at 2.5, 7.5, ... ms. This removes the
  // command/physics race at exact shared boundaries while keeping four 5 ms
  // steps in each 20 ms interval.
  std::uint64_t scheduled_ns =
      monotonic_ns_mujoco() - period_ns + period_ns / 2;
  CommandSnapshot command;
  command.joint_target = default_joint_position_;
  bool gains_damped = false;
  while (!stop_requested_.load(std::memory_order_relaxed)) {
    scheduled_ns += period_ns;
    const timespec target = mujoco_timespec(scheduled_ns);
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &target, nullptr) ==
           EINTR) {
    }
    if (stop_requested_.load(std::memory_order_relaxed)) {
      break;
    }
    const std::uint64_t woke_ns = monotonic_ns_mujoco();
    if (woke_ns > scheduled_ns) {
      update_mujoco_max(wake_late_ns_max_, woke_ns - scheduled_ns);
    }
    snapshot_command(command);
    const bool damped = damp_requested_.load(std::memory_order_acquire);
    if (damped != gains_damped) {
      if (damped) {
        std::array<float, kJointCount> zero{};
        std::array<float, kJointCount> damp;
        damp.fill(8.0F);
        set_gains(zero, damp);
      } else {
        set_gains(stiffness_, damping_);
      }
      gains_damped = damped;
    }
    for (std::size_t actuator = 0; actuator < kJointCount; ++actuator) {
      impl_->data->ctrl[actuator] =
          damped ? 0.0F : command.joint_target[actuator_to_isaac_[actuator]];
    }
    mj_step(impl_->model, impl_->data);
    physics_steps_.fetch_add(1, std::memory_order_relaxed);
    publish_state();
    const std::uint64_t finished_ns = monotonic_ns_mujoco();
    if (finished_ns > scheduled_ns + period_ns) {
      deadline_misses_.fetch_add(1, std::memory_order_relaxed);
    }
    if (physics_fault_.load(std::memory_order_acquire)) {
      break;
    }
  }
  running_.store(false, std::memory_order_release);
}

bool NativeMujocoBackend::healthy(double state_absent_ms) const noexcept {
  if (physics_fault_.load(std::memory_order_acquire)) {
    return false;
  }
  const std::uint64_t receive_ns =
      last_state_read_ns_.load(std::memory_order_relaxed);
  if (receive_ns == 0) {
    return false;
  }
  const double age_ms =
      static_cast<double>(monotonic_ns_mujoco() - receive_ns) / 1.0e6;
  return age_ms <= state_absent_ms;
}

NativeBackendTimingStats NativeMujocoBackend::timing_stats() const noexcept {
  return NativeBackendTimingStats{
      .steps = physics_steps_.load(std::memory_order_relaxed),
      .wake_late_ns_max = wake_late_ns_max_.load(std::memory_order_relaxed),
      .deadline_misses = deadline_misses_.load(std::memory_order_relaxed),
      .realtime_configured =
          realtime_configured_.load(std::memory_order_relaxed),
  };
}

}  // namespace ec_native
