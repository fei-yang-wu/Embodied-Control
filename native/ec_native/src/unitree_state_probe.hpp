#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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
  std::array<std::uint32_t, 29> motor_state{};
  std::array<std::uint32_t, 29> first_motor_error_code{};
  std::array<float, 4> quaternion{};
  std::array<float, 3> gyroscope{};
  std::array<float, 3> accelerometer{};
};

//: One captured LowState row: q[29] dq[29] tau[29] quat[4] gyro[3] accel[3].
constexpr std::size_t kProbeSampleWidth = 29 * 3 + 4 + 3 + 3;

class UnitreeStateProbe final {
 public:
  // `sample_capacity` > 0 keeps the first that many raw samples (values and
  // receive time) for offline sensor-noise analysis; the counters are
  // unaffected. The buffer is preallocated, so the handler never allocates.
  UnitreeStateProbe(const std::string& network_interface, int dds_domain = 0,
                    std::size_t sample_capacity = 0);
  ~UnitreeStateProbe();

  UnitreeStateProbe(const UnitreeStateProbe&) = delete;
  UnitreeStateProbe& operator=(const UnitreeStateProbe&) = delete;

  bool wait_for_samples(std::uint64_t count, double timeout_seconds) const;
  UnitreeProbeSnapshot snapshot() const noexcept;
  std::size_t captured_samples() const noexcept;
  std::size_t sample_capacity() const noexcept { return sample_capacity_; }
  const std::vector<float>& sample_values() const noexcept { return sample_values_; }
  const std::vector<std::uint64_t>& sample_times_ns() const noexcept { return sample_times_ns_; }
  const std::vector<std::uint32_t>& sample_ticks() const noexcept { return sample_ticks_; }

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
  std::array<std::atomic<std::uint32_t>, 29> first_motor_error_code_{};
  std::atomic<bool> tick_seen_{false};
  std::size_t sample_capacity_ = 0;
  std::atomic<std::uint64_t> captured_{0};
  std::vector<float> sample_values_;
  std::vector<std::uint64_t> sample_times_ns_;
  std::vector<std::uint32_t> sample_ticks_;
};

}  // namespace ec_native
