#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace ec_native {

struct OdometryProbeSnapshot {
  std::uint64_t frames = 0;
  std::uint64_t nonfinite_frames = 0;
  std::uint64_t first_receive_ns = 0;
  std::uint64_t last_receive_ns = 0;
  std::uint64_t max_gap_ns = 0;
  std::array<float, 3> first_position{};
  std::array<float, 3> position{};
  std::array<float, 3> velocity{};
  std::array<float, 4> quaternion_wxyz{};
  std::uint32_t error_code = 0;
  std::uint8_t mode = 0;
};

// Read-only listener on the G1's odometry topic (unitree_go
// SportModeState_): is the vendor's state estimator on the wire, at what
// rate, and does its position move. Use it before a live-anchor run on
// hardware, because the estimator may stop with the motion service the
// lifecycle releases; the live anchor then falls back to leg kinematics.
class OdometryProbe final {
 public:
  OdometryProbe(const std::string& network_interface,
                const std::string& topic = "rt/odommodestate",
                int dds_domain = 0);
  ~OdometryProbe();
  OdometryProbe(const OdometryProbe&) = delete;
  OdometryProbe& operator=(const OdometryProbe&) = delete;

  bool wait_for_frames(std::uint64_t count, double timeout_seconds) const;
  OdometryProbeSnapshot snapshot() const noexcept;

 private:
  struct Impl;
  struct Slot;
  void handler(const void* message) noexcept;
  std::unique_ptr<Impl> impl_;
  std::unique_ptr<Slot> slot_;
  std::atomic<std::uint64_t> frames_{0};
  std::atomic<std::uint64_t> nonfinite_frames_{0};
  std::atomic<std::uint64_t> first_receive_ns_{0};
  std::atomic<std::uint64_t> last_receive_ns_{0};
  std::atomic<std::uint64_t> max_gap_ns_{0};
};

}  // namespace ec_native
