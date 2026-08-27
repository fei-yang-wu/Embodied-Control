#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace ec_native {

struct UnitreeProbeSnapshot {
  std::uint64_t samples = 0;
  std::uint64_t crc_errors = 0;
  std::uint64_t nonfinite_samples = 0;
  std::uint64_t motor_error_samples = 0;
  std::uint64_t duplicate_ticks = 0;
  std::uint64_t first_receive_ns = 0;
  std::uint64_t last_receive_ns = 0;
  std::uint64_t max_gap_ns = 0;
  std::uint32_t first_tick = 0;
  std::uint32_t last_tick = 0;
  std::uint32_t motor_error_mask = 0;
  std::uint32_t mode_machine = 0;
  bool snapshot_consistent = false;
  std::array<float, 29> joint_position{};
  std::array<float, 29> joint_velocity{};
  std::array<float, 29> joint_torque{};
  std::array<float, 4> quaternion{};
  std::array<float, 3> gyroscope{};
  std::array<float, 3> accelerometer{};
};

class UnitreeStateProbe final {
 public:
  UnitreeStateProbe(const std::string& network_interface, int dds_domain = 0);
  ~UnitreeStateProbe();

  UnitreeStateProbe(const UnitreeStateProbe&) = delete;
  UnitreeStateProbe& operator=(const UnitreeStateProbe&) = delete;

  bool wait_for_samples(std::uint64_t count, double timeout_seconds) const;
  UnitreeProbeSnapshot snapshot() const noexcept;

 private:
  struct Impl;
  struct Slot;

  void low_state_handler(const void* message) noexcept;

  std::unique_ptr<Impl> impl_;
  std::unique_ptr<Slot> slot_;
  std::atomic<std::uint64_t> samples_{0};
  std::atomic<std::uint64_t> crc_errors_{0};
  std::atomic<std::uint64_t> nonfinite_samples_{0};
  std::atomic<std::uint64_t> motor_error_samples_{0};
  std::atomic<std::uint64_t> duplicate_ticks_{0};
  std::atomic<std::uint64_t> first_receive_ns_{0};
  std::atomic<std::uint64_t> last_receive_ns_{0};
  std::atomic<std::uint64_t> max_gap_ns_{0};
  std::atomic<std::uint32_t> first_tick_{0};
  std::atomic<std::uint32_t> last_tick_{0};
  std::atomic<std::uint32_t> motor_error_mask_{0};
  std::atomic<bool> tick_seen_{false};
};

}  // namespace ec_native
