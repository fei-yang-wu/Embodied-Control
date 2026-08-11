#include "mujoco_dds_plant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>

#include <mujoco/mujoco.h>

#include <unitree/dds_wrapper/common/crc.h>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "native_backend.hpp"

namespace ec_native {
namespace {

using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;
using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelPublisherPtr;
using unitree::robot::ChannelSubscriber;
using unitree::robot::ChannelSubscriberPtr;

constexpr int kPlantSnapshotAttempts = 8;
constexpr double kGravity = 9.81;

std::uint64_t monotonic_ns_plant() noexcept {
  timespec now{};
  clock_gettime(CLOCK_MONOTONIC, &now);
  return static_cast<std::uint64_t>(now.tv_sec) * 1000000000ULL +
         static_cast<std::uint64_t>(now.tv_nsec);
}

timespec plant_timespec(std::uint64_t nanoseconds) noexcept {
  return timespec{
      static_cast<time_t>(nanoseconds / 1000000000ULL),
      static_cast<long>(nanoseconds % 1000000000ULL),
  };
}

void update_plant_max(std::atomic<std::uint64_t>& destination,
                      std::uint64_t value) noexcept {
  std::uint64_t previous = destination.load(std::memory_order_relaxed);
  while (previous < value &&
         !destination.compare_exchange_weak(previous, value,
                                            std::memory_order_relaxed)) {
  }
}

bool finite_plant_values(std::span<const float> values) {
  return std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); });
}

}  // namespace

struct MujocoDdsPlant::Impl {
  mjModel* model = nullptr;
  mjData* data = nullptr;
  ChannelPublisherPtr<LowState> publisher;
  ChannelSubscriberPtr<LowCmd> subscriber;
  LowState state_message{};
  std::array<float, kJointCount> applied_stiffness{};
  std::array<float, kJointCount> applied_damping{};

  ~Impl() {
    subscriber.reset();
    publisher.reset();
    if (data != nullptr) {
      mj_deleteData(data);
    }
    if (model != nullptr) {
      mj_deleteModel(model);
    }
  }
};

struct MujocoDdsPlant::CommandSlot {
  std::atomic<std::uint64_t> sequence{0};
  std::array<std::atomic<float>, kJointCount> q{};
  std::array<std::atomic<float>, kJointCount> dq{};
  std::array<std::atomic<float>, kJointCount> tau{};
  std::array<std::atomic<float>, kJointCount> kp{};
  std::array<std::atomic<float>, kJointCount> kd{};
  std::atomic<std::uint64_t> receive_ns{0};
  std::atomic<bool> valid{false};
};

struct MujocoDdsPlant::CommandSnapshot {
  std::array<float, kJointCount> q{};
  std::array<float, kJointCount> dq{};
  std::array<float, kJointCount> tau{};
  std::array<float, kJointCount> kp{};
  std::array<float, kJointCount> kd{};
  std::uint64_t receive_ns = 0;
  bool valid = false;
};

MujocoDdsPlant::MujocoDdsPlant(
    const std::string& model_path, const std::string& network_interface,
    const std::vector<std::string>& sdk_joint_names,
    std::span<const float> default_joint_position,
    std::span<const float> armature, std::span<const float> effort_limit,
    std::span<const float> hold_stiffness, std::span<const float> hold_damping,
    double timestep, std::uint8_t mode_machine, int physics_cpu,
    int physics_fifo_priority, bool lock_memory, bool require_realtime)
    : impl_(std::make_unique<Impl>()),
      command_slot_(std::make_unique<CommandSlot>()),
      timestep_(timestep),
      mode_machine_(mode_machine),
      physics_cpu_(physics_cpu),
      physics_fifo_priority_(physics_fifo_priority),
      lock_memory_(lock_memory),
      require_realtime_(require_realtime) {
  if (network_interface.empty() || sdk_joint_names.size() != kJointCount ||
      default_joint_position.size() != kJointCount ||
      armature.size() != kJointCount || effort_limit.size() != kJointCount ||
      hold_stiffness.size() != kJointCount ||
      hold_damping.size() != kJointCount ||
      !finite_plant_values(default_joint_position) ||
      !finite_plant_values(armature) || !finite_plant_values(effort_limit) ||
      !finite_plant_values(hold_stiffness) ||
      !finite_plant_values(hold_damping) ||
      std::any_of(armature.begin(), armature.end(),
                  [](float value) { return value < 0.0F; }) ||
      std::any_of(effort_limit.begin(), effort_limit.end(),
                  [](float value) { return value <= 0.0F; }) ||
      std::any_of(hold_stiffness.begin(), hold_stiffness.end(),
                  [](float value) { return value < 0.0F; }) ||
      std::any_of(hold_damping.begin(), hold_damping.end(),
                  [](float value) { return value < 0.0F; }) ||
      !std::isfinite(timestep_) || timestep_ <= 0.0 || physics_cpu_ < -1 ||
      physics_cpu_ >= CPU_SETSIZE || physics_fifo_priority_ < 0 ||
      physics_fifo_priority_ > sched_get_priority_max(SCHED_FIFO)) {
    throw std::runtime_error("invalid MuJoCo DDS plant configuration");
  }
  std::copy(default_joint_position.begin(), default_joint_position.end(),
            default_joint_position_.begin());
  std::copy(effort_limit.begin(), effort_limit.end(), effort_limit_.begin());
  std::copy(hold_stiffness.begin(), hold_stiffness.end(),
            hold_stiffness_.begin());
  std::copy(hold_damping.begin(), hold_damping.end(), hold_damping_.begin());

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
  if (model->nq < 7 || model->nv < 6) {
    throw std::runtime_error("MuJoCo model has no floating base");
  }
  model->opt.timestep = timestep_;
  model->opt.integrator = mjINT_IMPLICITFAST;
  if (model->nu != static_cast<int>(kJointCount)) {
    throw std::runtime_error("MuJoCo model must have 29 actuators");
  }
  sdk_to_actuator_.fill(-1);
  for (std::size_t actuator = 0; actuator < kJointCount; ++actuator) {
    const int joint = model->actuator_trnid[2 * actuator];
    const char* joint_name = mj_id2name(model, mjOBJ_JOINT, joint);
    if (joint_name == nullptr) {
      throw std::runtime_error("MuJoCo actuator joint has no name");
    }
    const auto found =
        std::find(sdk_joint_names.begin(), sdk_joint_names.end(),
                  std::string(joint_name));
    if (found == sdk_joint_names.end()) {
      throw std::runtime_error("MuJoCo joint is absent from the SDK table: " +
                               std::string(joint_name));
    }
    const std::size_t sdk =
        static_cast<std::size_t>(found - sdk_joint_names.begin());
    actuator_to_sdk_[actuator] = sdk;
    sdk_to_actuator_[sdk] = static_cast<int>(actuator);
    qpos_address_[actuator] = model->jnt_qposadr[joint];
    const int dof = model->jnt_dofadr[joint];
    dof_address_[actuator] = dof;
    model->dof_armature[dof] = armature[sdk];
    model->dof_damping[dof] = 0.0;
    model->dof_frictionloss[dof] = 0.0;
    model->actuator_forcelimited[actuator] = 1;
    model->actuator_forcerange[2 * actuator] = -effort_limit[sdk];
    model->actuator_forcerange[2 * actuator + 1] = effort_limit[sdk];
    model->actuator_ctrllimited[actuator] = 0;
  }
  if (std::any_of(sdk_to_actuator_.begin(), sdk_to_actuator_.end(),
                  [](int value) { return value < 0; })) {
    throw std::runtime_error("sdk_joint_names must cover all 29 actuators");
  }
  reset();

  ChannelFactory::Instance()->Init(0, network_interface);
  impl_->publisher =
      std::make_shared<ChannelPublisher<LowState>>("rt/lowstate");
  impl_->publisher->InitChannel();
  impl_->subscriber = std::make_shared<ChannelSubscriber<LowCmd>>("rt/lowcmd");
  impl_->subscriber->InitChannel(
      std::bind(&MujocoDdsPlant::low_cmd_handler, this, std::placeholders::_1),
      1);
}

MujocoDdsPlant::~MujocoDdsPlant() {
  stop();
  wait_for_stop();
  impl_->subscriber.reset();
}

void MujocoDdsPlant::set_initial_pose(std::span<const float> pose) {
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

void MujocoDdsPlant::set_servo_gains(std::span<const float> stiffness,
                                     std::span<const float> damping) noexcept {
  mjModel* model = impl_->model;
  for (std::size_t actuator = 0; actuator < kJointCount; ++actuator) {
    const std::size_t sdk = actuator_to_sdk_[actuator];
    model->actuator_gaintype[actuator] = mjGAIN_FIXED;
    model->actuator_biastype[actuator] = mjBIAS_AFFINE;
    mju_zero(model->actuator_gainprm + actuator * mjNGAIN, mjNGAIN);
    mju_zero(model->actuator_biasprm + actuator * mjNBIAS, mjNBIAS);
    model->actuator_gainprm[actuator * mjNGAIN] = stiffness[sdk];
    model->actuator_biasprm[actuator * mjNBIAS + 1] = -stiffness[sdk];
    model->actuator_biasprm[actuator * mjNBIAS + 2] = -damping[sdk];
    impl_->applied_stiffness[sdk] = stiffness[sdk];
    impl_->applied_damping[sdk] = damping[sdk];
  }
}

void MujocoDdsPlant::reset() {
  if (running_.load(std::memory_order_acquire)) {
    throw std::runtime_error("stop the plant before reset");
  }
  wait_for_stop();
  mj_resetData(impl_->model, impl_->data);
  mjData* data = impl_->data;
  if (has_initial_pose_) {
    data->qpos[0] = initial_pose_[0];
    data->qpos[1] = initial_pose_[1];
    data->qpos[2] = initial_pose_[2];
    // The pose contract carries XYZW; MuJoCo stores WXYZ.
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
    const std::size_t sdk = actuator_to_sdk_[actuator];
    const float joint = has_initial_pose_ ? initial_pose_[7 + sdk]
                                          : default_joint_position_[sdk];
    data->qpos[qpos_address_[actuator]] = joint;
    data->ctrl[actuator] = joint;
    data->qfrc_applied[dof_address_[actuator]] = 0.0;
  }
  set_servo_gains(hold_stiffness_, hold_damping_);
  mj_forward(impl_->model, data);
  data->time = 0.0;
  command_slot_->sequence.store(0, std::memory_order_relaxed);
  command_slot_->valid.store(false, std::memory_order_relaxed);
  command_slot_->receive_ns.store(0, std::memory_order_relaxed);
  stop_requested_.store(false, std::memory_order_relaxed);
  thread_ready_.store(false, std::memory_order_relaxed);
  realtime_configured_.store(false, std::memory_order_relaxed);
  physics_fault_.store(false, std::memory_order_relaxed);
  holding_.store(true, std::memory_order_relaxed);
  steps_.store(0, std::memory_order_relaxed);
  publishes_.store(0, std::memory_order_relaxed);
  publish_failures_.store(0, std::memory_order_relaxed);
  crc_errors_.store(0, std::memory_order_relaxed);
  wake_late_ns_max_.store(0, std::memory_order_relaxed);
  deadline_misses_.store(0, std::memory_order_relaxed);
  last_command_ns_.store(0, std::memory_order_relaxed);
  simulation_time_.store(0.0, std::memory_order_relaxed);
  base_height_.store(static_cast<float>(data->qpos[2]),
                     std::memory_order_relaxed);
  min_base_height_.store(static_cast<float>(data->qpos[2]),
                         std::memory_order_relaxed);
}

void MujocoDdsPlant::low_cmd_handler(const void* message) noexcept {
  const LowCmd& input = *static_cast<const LowCmd*>(message);
  if (input.crc() !=
      crc32_core(reinterpret_cast<std::uint32_t*>(const_cast<LowCmd*>(&input)),
                 (sizeof(LowCmd) >> 2) - 1)) {
    crc_errors_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  for (std::size_t sdk = 0; sdk < kJointCount; ++sdk) {
    const auto& motor = input.motor_cmd()[sdk];
    // A frame with non-finite or negative-gain motor values would poison the
    // integrator state permanently, so it is rejected like a corrupt frame.
    if (!std::isfinite(motor.q()) || !std::isfinite(motor.dq()) ||
        !std::isfinite(motor.tau()) || !std::isfinite(motor.kp()) ||
        !std::isfinite(motor.kd()) || motor.kp() < 0.0F || motor.kd() < 0.0F) {
      crc_errors_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  }
  const std::uint64_t sequence =
      command_slot_->sequence.load(std::memory_order_relaxed);
  command_slot_->sequence.store(sequence + 1, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  for (std::size_t sdk = 0; sdk < kJointCount; ++sdk) {
    const auto& motor = input.motor_cmd()[sdk];
    command_slot_->q[sdk].store(motor.q(), std::memory_order_relaxed);
    command_slot_->dq[sdk].store(motor.dq(), std::memory_order_relaxed);
    command_slot_->tau[sdk].store(motor.tau(), std::memory_order_relaxed);
    command_slot_->kp[sdk].store(motor.kp(), std::memory_order_relaxed);
    command_slot_->kd[sdk].store(motor.kd(), std::memory_order_relaxed);
  }
  const std::uint64_t now_ns = monotonic_ns_plant();
  command_slot_->receive_ns.store(now_ns, std::memory_order_relaxed);
  command_slot_->valid.store(true, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  command_slot_->sequence.store(sequence + 2, std::memory_order_relaxed);
  last_command_ns_.store(now_ns, std::memory_order_relaxed);
  commands_received_.fetch_add(1, std::memory_order_relaxed);
}

bool MujocoDdsPlant::snapshot_command(CommandSnapshot& destination) const
    noexcept {
  for (int attempt = 0; attempt < kPlantSnapshotAttempts; ++attempt) {
    const std::uint64_t first =
        command_slot_->sequence.load(std::memory_order_acquire);
    if ((first & 1ULL) != 0) {
      continue;
    }
    for (std::size_t sdk = 0; sdk < kJointCount; ++sdk) {
      destination.q[sdk] = command_slot_->q[sdk].load(std::memory_order_relaxed);
      destination.dq[sdk] =
          command_slot_->dq[sdk].load(std::memory_order_relaxed);
      destination.tau[sdk] =
          command_slot_->tau[sdk].load(std::memory_order_relaxed);
      destination.kp[sdk] =
          command_slot_->kp[sdk].load(std::memory_order_relaxed);
      destination.kd[sdk] =
          command_slot_->kd[sdk].load(std::memory_order_relaxed);
    }
    destination.receive_ns =
        command_slot_->receive_ns.load(std::memory_order_relaxed);
    destination.valid = command_slot_->valid.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (first == command_slot_->sequence.load(std::memory_order_relaxed)) {
      return destination.valid;
    }
  }
  return false;
}

void MujocoDdsPlant::start() {
  if (running_.load(std::memory_order_acquire) || physics_thread_.joinable()) {
    throw std::runtime_error("the plant physics thread is already started");
  }
  stop_requested_.store(false, std::memory_order_relaxed);
  thread_ready_.store(false, std::memory_order_relaxed);
  running_.store(true, std::memory_order_release);
  physics_thread_ = std::thread(&MujocoDdsPlant::physics_loop, this);
  while (!thread_ready_.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  if (require_realtime_ &&
      !realtime_configured_.load(std::memory_order_acquire)) {
    stop();
    wait_for_stop();
    throw std::runtime_error(
        "the plant physics thread did not obtain required real-time settings");
  }
}

void MujocoDdsPlant::stop() noexcept {
  stop_requested_.store(true, std::memory_order_release);
}

void MujocoDdsPlant::wait_for_stop() noexcept {
  if (physics_thread_.joinable()) {
    physics_thread_.join();
  }
}

bool MujocoDdsPlant::configure_physics_thread() noexcept {
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

void MujocoDdsPlant::publish_low_state() noexcept {
  const mjData* data = impl_->data;
  LowState& message = impl_->state_message;
  message.mode_machine() = mode_machine_;
  message.tick() = static_cast<std::uint32_t>(
      static_cast<double>(steps_.load(std::memory_order_relaxed)) * timestep_ *
      1000.0);
  bool valid = std::isfinite(data->time) && std::isfinite(data->qpos[2]);
  for (std::size_t actuator = 0; actuator < kJointCount; ++actuator) {
    const std::size_t sdk = actuator_to_sdk_[actuator];
    auto& motor = message.motor_state()[sdk];
    const float position =
        static_cast<float>(data->qpos[qpos_address_[actuator]]);
    const float velocity =
        static_cast<float>(data->qvel[dof_address_[actuator]]);
    motor.q() = position;
    motor.dq() = velocity;
    motor.tau_est() =
        static_cast<float>(data->actuator_force[actuator] +
                           data->qfrc_applied[dof_address_[actuator]]);
    valid = valid && std::isfinite(position) && std::isfinite(velocity);
  }
  auto& imu = message.imu_state();
  const std::array<float, 4> quaternion_wxyz = {
      static_cast<float>(data->qpos[3]),
      static_cast<float>(data->qpos[4]),
      static_cast<float>(data->qpos[5]),
      static_cast<float>(data->qpos[6]),
  };
  for (std::size_t index = 0; index < 4; ++index) {
    imu.quaternion()[index] = quaternion_wxyz[index];
    valid = valid && std::isfinite(quaternion_wxyz[index]);
  }
  for (std::size_t index = 0; index < 3; ++index) {
    const float value = static_cast<float>(data->qvel[3 + index]);
    imu.gyroscope()[index] = value;
    valid = valid && std::isfinite(value);
  }
  const std::array<float, 4> quaternion_xyzw = {
      quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3],
      quaternion_wxyz[0]};
  std::array<float, 3> gravity_body{0.0F, 0.0F, -1.0F};
  if (projected_gravity_from_xyzw(quaternion_xyzw, gravity_body)) {
    for (std::size_t index = 0; index < 3; ++index) {
      imu.accelerometer()[index] =
          static_cast<float>(-kGravity) * gravity_body[index];
    }
  }
  message.crc() = crc32_core(reinterpret_cast<std::uint32_t*>(&message),
                             (sizeof(message) >> 2) - 1);
  bool published = false;
  try {
    published = impl_->publisher->Write(message);
  } catch (...) {
  }
  if (published) {
    publishes_.fetch_add(1, std::memory_order_relaxed);
  } else {
    publish_failures_.fetch_add(1, std::memory_order_relaxed);
  }
  simulation_time_.store(data->time, std::memory_order_relaxed);
  const float height = static_cast<float>(data->qpos[2]);
  base_height_.store(height, std::memory_order_relaxed);
  float previous_min = min_base_height_.load(std::memory_order_relaxed);
  while (height < previous_min &&
         !min_base_height_.compare_exchange_weak(previous_min, height,
                                                 std::memory_order_relaxed)) {
  }
  if (!valid) {
    physics_fault_.store(true, std::memory_order_release);
  }
}

void MujocoDdsPlant::physics_loop() noexcept {
  const bool configured = configure_physics_thread();
  thread_ready_.store(true, std::memory_order_release);
  if (!configured && require_realtime_) {
    physics_fault_.store(true, std::memory_order_release);
    running_.store(false, std::memory_order_release);
    return;
  }
  const auto period_ns =
      static_cast<std::uint64_t>(std::llround(timestep_ * 1000000000.0));
  std::uint64_t scheduled_ns = monotonic_ns_plant();
  CommandSnapshot command;
  while (!stop_requested_.load(std::memory_order_relaxed)) {
    scheduled_ns += period_ns;
    const timespec target = plant_timespec(scheduled_ns);
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &target, nullptr) ==
           EINTR) {
    }
    if (stop_requested_.load(std::memory_order_relaxed)) {
      break;
    }
    const std::uint64_t woke_ns = monotonic_ns_plant();
    if (woke_ns > scheduled_ns) {
      update_plant_max(wake_late_ns_max_, woke_ns - scheduled_ns);
    }
    if (snapshot_command(command)) {
      holding_.store(false, std::memory_order_relaxed);
      if (std::memcmp(command.kp.data(), impl_->applied_stiffness.data(),
                      sizeof(command.kp)) != 0 ||
          std::memcmp(command.kd.data(), impl_->applied_damping.data(),
                      sizeof(command.kd)) != 0) {
        set_servo_gains(command.kp, command.kd);
      }
      float kp_min = command.kp[0];
      float kp_max = command.kp[0];
      float kd_min = command.kd[0];
      float kd_max = command.kd[0];
      float q_absmax = applied_q_absmax_.load(std::memory_order_relaxed);
      float extra_absmax =
          applied_extra_absmax_.load(std::memory_order_relaxed);
      for (std::size_t actuator = 0; actuator < kJointCount; ++actuator) {
        const std::size_t sdk = actuator_to_sdk_[actuator];
        impl_->data->ctrl[actuator] = command.q[sdk];
        // The servo term covers kp (q_des - q) - kd dq; the firmware law's
        // remaining kd dq_des + tau_ff part is injected as an applied force.
        const float extra =
            command.tau[sdk] + command.kd[sdk] * command.dq[sdk];
        impl_->data->qfrc_applied[dof_address_[actuator]] = std::clamp(
            extra, -effort_limit_[sdk], effort_limit_[sdk]);
        kp_min = std::min(kp_min, command.kp[sdk]);
        kp_max = std::max(kp_max, command.kp[sdk]);
        kd_min = std::min(kd_min, command.kd[sdk]);
        kd_max = std::max(kd_max, command.kd[sdk]);
        q_absmax = std::max(q_absmax, std::abs(command.q[sdk]));
        extra_absmax = std::max(extra_absmax, std::abs(extra));
      }
      applied_kp_min_.store(kp_min, std::memory_order_relaxed);
      applied_kp_max_.store(kp_max, std::memory_order_relaxed);
      applied_kd_min_.store(kd_min, std::memory_order_relaxed);
      applied_kd_max_.store(kd_max, std::memory_order_relaxed);
      applied_q_absmax_.store(q_absmax, std::memory_order_relaxed);
      applied_extra_absmax_.store(extra_absmax, std::memory_order_relaxed);
    }
    mj_step(impl_->model, impl_->data);
    steps_.fetch_add(1, std::memory_order_relaxed);
    publish_low_state();
    const std::uint64_t finished_ns = monotonic_ns_plant();
    if (finished_ns > scheduled_ns + period_ns) {
      deadline_misses_.fetch_add(1, std::memory_order_relaxed);
    }
    if (physics_fault_.load(std::memory_order_acquire)) {
      break;
    }
  }
  running_.store(false, std::memory_order_release);
}

PlantStats MujocoDdsPlant::stats() const noexcept {
  const std::uint64_t last_command_ns =
      last_command_ns_.load(std::memory_order_relaxed);
  return PlantStats{
      .steps = steps_.load(std::memory_order_relaxed),
      .publishes = publishes_.load(std::memory_order_relaxed),
      .publish_failures = publish_failures_.load(std::memory_order_relaxed),
      .commands_received = commands_received_.load(std::memory_order_relaxed),
      .crc_errors = crc_errors_.load(std::memory_order_relaxed),
      .wake_late_ns_max = wake_late_ns_max_.load(std::memory_order_relaxed),
      .deadline_misses = deadline_misses_.load(std::memory_order_relaxed),
      .holding = holding_.load(std::memory_order_relaxed),
      .physics_fault = physics_fault_.load(std::memory_order_relaxed),
      .realtime_configured =
          realtime_configured_.load(std::memory_order_relaxed),
      .last_command_age_ms =
          last_command_ns == 0
              ? -1.0
              : static_cast<double>(monotonic_ns_plant() - last_command_ns) /
                    1.0e6,
      .time = simulation_time_.load(std::memory_order_relaxed),
      .base_height =
          static_cast<double>(base_height_.load(std::memory_order_relaxed)),
      .min_base_height = static_cast<double>(
          min_base_height_.load(std::memory_order_relaxed)),
      .applied_kp_min = static_cast<double>(
          applied_kp_min_.load(std::memory_order_relaxed)),
      .applied_kp_max = static_cast<double>(
          applied_kp_max_.load(std::memory_order_relaxed)),
      .applied_kd_min = static_cast<double>(
          applied_kd_min_.load(std::memory_order_relaxed)),
      .applied_kd_max = static_cast<double>(
          applied_kd_max_.load(std::memory_order_relaxed)),
      .applied_q_absmax = static_cast<double>(
          applied_q_absmax_.load(std::memory_order_relaxed)),
      .applied_extra_absmax = static_cast<double>(
          applied_extra_absmax_.load(std::memory_order_relaxed)),
  };
}

}  // namespace ec_native
