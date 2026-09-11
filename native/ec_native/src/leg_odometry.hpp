#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ec_native {

// Kinematic leg odometry for the G1: pelvis translation in the world from
// the joint positions and the IMU orientation alone, no contact sensor.
//
// The contact set is every sole point (four per foot) within
// `contact_threshold_m` of the lowest sole point, in both the previous and
// the current frame. Those points are assumed planted, so the pelvis moved
// by minus the per-axis median of their world-frame displacements. The
// median rejects the points of a foot that is rolling on its edge or
// sliding; the mean of a single "stance" foot picked by height drifted
// 5-9 % of the path on the plant, the contact-set median about 1 %. Pelvis
// height is the height of the pelvis above the lowest sole point.
//
// The estimate is the same kind the vendor's state estimator serves on
// `rt/odommodestate`: a random walk whose step error is the forward
// kinematics error over one control tick. The tracker only needs the error
// between the robot and its reference over the encoder window, so the walk
// is bounded by the odometry drift, not by the reference's displacement
// (the frozen start anchor's failure mode).
class LegOdometry {
 public:
  // `mjcf_path` is the G1 model with `pelvis`, `left_ankle_roll_link` and
  // `right_ankle_roll_link` bodies. `isaac_joint_names` gives the order of
  // the joint positions passed to `update`, mapped to the model's joints by
  // name.
  LegOdometry(const std::string& mjcf_path,
              std::span<const std::string> isaac_joint_names,
              float contact_threshold_m = 0.005F);
  ~LegOdometry();
  LegOdometry(const LegOdometry&) = delete;
  LegOdometry& operator=(const LegOdometry&) = delete;

  // Forget the walk: the next update restarts at (0, 0, height above the
  // lowest sole point).
  void reset() noexcept;

  // One control tick. `quaternion_xyzw` is the pelvis IMU orientation in
  // the world. Returns false (and leaves `position_w` untouched) on a
  // non-finite input. Never allocates after construction.
  bool update(std::span<const float> joint_position,
              std::span<const float> quaternion_xyzw,
              std::array<float, 3>& position_w) noexcept;

  // The foot that owns most of the contact set: 0 = left, 1 = right.
  int stance_foot() const noexcept { return stance_foot_; }
  std::uint64_t stance_switches() const noexcept { return stance_switches_; }
  std::uint64_t updates() const noexcept { return updates_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::vector<int> qpos_address_;
  std::array<int, 2> foot_body_{-1, -1};
  // Sole reference points in the ankle-roll body frame (the contact
  // spheres in the vendor MJCF sit at z = -0.03 with radius 0.005).
  static constexpr std::array<std::array<double, 3>, 4> kSolePoints = {{
      {-0.05, 0.025, -0.035},
      {-0.05, -0.025, -0.035},
      {0.12, 0.03, -0.035},
      {0.12, -0.03, -0.035},
  }};
  static constexpr std::size_t kPointCount = 2 * kSolePoints.size();
  float contact_threshold_m_;
  int stance_foot_ = -1;
  std::array<double, 3> position_w_{0.0, 0.0, 0.0};
  std::array<std::array<double, 3>, kPointCount> previous_points_w_{};
  double previous_lowest_ = 0.0;
  bool primed_ = false;
  std::uint64_t stance_switches_ = 0;
  std::uint64_t updates_ = 0;
};

}  // namespace ec_native
