#include "leg_odometry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <mujoco/mujoco.h>

namespace ec_native {

struct LegOdometry::Impl {
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

LegOdometry::LegOdometry(const std::string& mjcf_path,
                         std::span<const std::string> isaac_joint_names,
                         float contact_threshold_m)
    : impl_(std::make_unique<Impl>()),
      contact_threshold_m_(contact_threshold_m) {
  if (!(contact_threshold_m_ > 0.0F)) {
    throw std::runtime_error("contact threshold must be positive");
  }
  char error[1024] = {0};
  impl_->model = mj_loadXML(mjcf_path.c_str(), nullptr, error, sizeof(error));
  if (impl_->model == nullptr) {
    throw std::runtime_error(std::string("leg odometry MJCF load failed: ") +
                             error);
  }
  impl_->data = mj_makeData(impl_->model);
  if (impl_->data == nullptr) {
    throw std::runtime_error("leg odometry MuJoCo data allocation failed");
  }
  mjModel* model = impl_->model;
  if (model->nq < 7 || model->jnt_type[0] != mjJNT_FREE) {
    throw std::runtime_error("leg odometry model needs a free base joint first");
  }
  qpos_address_.resize(isaac_joint_names.size());
  for (std::size_t index = 0; index < isaac_joint_names.size(); ++index) {
    const int joint =
        mj_name2id(model, mjOBJ_JOINT, isaac_joint_names[index].c_str());
    if (joint < 0) {
      throw std::runtime_error("leg odometry model has no joint named " +
                               isaac_joint_names[index]);
    }
    qpos_address_[index] = model->jnt_qposadr[joint];
  }
  foot_body_[0] = mj_name2id(model, mjOBJ_BODY, "left_ankle_roll_link");
  foot_body_[1] = mj_name2id(model, mjOBJ_BODY, "right_ankle_roll_link");
  if (foot_body_[0] < 0 || foot_body_[1] < 0) {
    throw std::runtime_error(
        "leg odometry model needs left_ankle_roll_link and "
        "right_ankle_roll_link");
  }
  if (mj_name2id(model, mjOBJ_BODY, "pelvis") < 0) {
    throw std::runtime_error("leg odometry model has no pelvis body");
  }
  reset();
}

LegOdometry::~LegOdometry() = default;

void LegOdometry::reset() noexcept {
  primed_ = false;
  stance_foot_ = -1;
  position_w_ = {0.0, 0.0, 0.0};
}

namespace {

// Median of the first `count` values, in place (count <= 8).
double median_in_place(std::array<double, 8>& values, std::size_t count) {
  std::sort(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(count));
  if (count % 2 == 1) {
    return values[count / 2];
  }
  return 0.5 * (values[count / 2 - 1] + values[count / 2]);
}

}  // namespace

bool LegOdometry::update(std::span<const float> joint_position,
                         std::span<const float> quaternion_xyzw,
                         std::array<float, 3>& position_w) noexcept {
  if (joint_position.size() != qpos_address_.size() ||
      quaternion_xyzw.size() != 4) {
    return false;
  }
  double norm_squared = 0.0;
  for (const float value : quaternion_xyzw) {
    if (!std::isfinite(value)) {
      return false;
    }
    norm_squared += static_cast<double>(value) * value;
  }
  if (!(norm_squared > 0.25 && norm_squared < 2.25)) {
    return false;
  }
  for (const float value : joint_position) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  mjModel* model = impl_->model;
  mjData* data = impl_->data;
  // Base at the origin with the IMU orientation: body positions then come
  // out as world-frame offsets from the pelvis.
  const double inverse_norm = 1.0 / std::sqrt(norm_squared);
  data->qpos[0] = 0.0;
  data->qpos[1] = 0.0;
  data->qpos[2] = 0.0;
  data->qpos[3] = quaternion_xyzw[3] * inverse_norm;  // w
  data->qpos[4] = quaternion_xyzw[0] * inverse_norm;  // x
  data->qpos[5] = quaternion_xyzw[1] * inverse_norm;  // y
  data->qpos[6] = quaternion_xyzw[2] * inverse_norm;  // z
  for (std::size_t index = 0; index < qpos_address_.size(); ++index) {
    data->qpos[qpos_address_[index]] = joint_position[index];
  }
  mj_kinematics(model, data);

  std::array<std::array<double, 3>, kPointCount> points_w{};
  double lowest = 1e9;
  for (int foot = 0; foot < 2; ++foot) {
    const int body = foot_body_[foot];
    const double* origin = data->xpos + 3 * body;
    const double* rotation = data->xmat + 9 * body;
    for (std::size_t corner = 0; corner < kSolePoints.size(); ++corner) {
      const auto& point = kSolePoints[corner];
      auto& world = points_w[foot * kSolePoints.size() + corner];
      for (int axis = 0; axis < 3; ++axis) {
        world[axis] = origin[axis] + rotation[3 * axis] * point[0] +
                      rotation[3 * axis + 1] * point[1] +
                      rotation[3 * axis + 2] * point[2];
      }
      lowest = std::min(lowest, world[2]);
    }
  }

  const double threshold = static_cast<double>(contact_threshold_m_);
  if (primed_) {
    // Planted now and planted before: a point that just touched down or
    // just lifted moved with the swing leg for part of the tick.
    std::array<double, 8> dx{}, dy{};
    std::size_t count = 0;
    std::array<int, 2> per_foot{0, 0};
    for (std::size_t point = 0; point < kPointCount; ++point) {
      if (points_w[point][2] <= lowest + threshold &&
          previous_points_w_[point][2] <= previous_lowest_ + threshold) {
        dx[count] = points_w[point][0] - previous_points_w_[point][0];
        dy[count] = points_w[point][1] - previous_points_w_[point][1];
        ++count;
        ++per_foot[point / kSolePoints.size()];
      }
    }
    if (count == 0) {
      for (std::size_t point = 0; point < kPointCount; ++point) {
        if (points_w[point][2] <= lowest + threshold) {
          dx[count] = points_w[point][0] - previous_points_w_[point][0];
          dy[count] = points_w[point][1] - previous_points_w_[point][1];
          ++count;
          ++per_foot[point / kSolePoints.size()];
        }
      }
    }
    if (count > 0) {
      // The planted points did not move; whatever changed in their
      // pelvis-relative position is pelvis motion in the opposite direction.
      position_w_[0] -= median_in_place(dx, count);
      position_w_[1] -= median_in_place(dy, count);
    }
    const int stance = per_foot[0] >= per_foot[1] ? 0 : 1;
    if (stance != stance_foot_) {
      ++stance_switches_;
      stance_foot_ = stance;
    }
  } else {
    primed_ = true;
    position_w_[0] = 0.0;
    position_w_[1] = 0.0;
    int left = 0, right = 0;
    for (std::size_t point = 0; point < kPointCount; ++point) {
      if (points_w[point][2] <= lowest + threshold) {
        (point < kSolePoints.size() ? left : right) += 1;
      }
    }
    stance_foot_ = left >= right ? 0 : 1;
  }
  position_w_[2] = -lowest;
  previous_points_w_ = points_w;
  previous_lowest_ = lowest;
  ++updates_;
  position_w[0] = static_cast<float>(position_w_[0]);
  position_w[1] = static_cast<float>(position_w_[1]);
  position_w[2] = static_cast<float>(position_w_[2]);
  return true;
}

}  // namespace ec_native
