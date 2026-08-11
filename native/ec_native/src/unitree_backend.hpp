#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <thread>

#include "native_backend.hpp"

namespace ec_native {

enum class UnitreeMode : std::uint32_t {
  kDisabled = 0,
  kInitialize = 1,
  kWait = 2,
  kControl = 3,
  kDamp = 4,
};

struct UnitreeWriterStats {
  std::uint64_t ticks = 0;
  std::uint64_t publishes = 0;
  std::uint64_t publish_failures = 0;
  std::uint64_t crc_errors = 0;
  std::uint64_t hardware_faults = 0;
  std::uint64_t watchdog_faults = 0;
  std::uint64_t wake_late_ns_max = 0;
  std::uint64_t deadline_misses = 0;
  UnitreeMode mode = UnitreeMode::kDisabled;
  bool writes_enabled = false;
  bool realtime_configured = false;
};

// Native Unitree G1 DDS backend. DDS receive and the 500 Hz command writer
// stay outside Python. Writes are disabled by default and need an explicit
// initialization plus arm sequence.
class NativeUnitreeBackend final : public NativeRobotBackend {
 public:
  NativeUnitreeBackend(
      const std::string& network_interface,
      std::span<const std::size_t> isaac_to_sdk,
      std::span<const float> default_joint_position,
      std::span<const float> stiffness, std::span<const float> damping,
      std::span<const float> joint_lower, std::span<const float> joint_upper,
      bool writes_enabled, int writer_cpu, int writer_fifo_priority,
      bool lock_memory, bool require_realtime, double state_absent_ms,
      double command_stale_ms);
  ~NativeUnitreeBackend() override;

  NativeUnitreeBackend(const NativeUnitreeBackend&) = delete;
  NativeUnitreeBackend& operator=(const NativeUnitreeBackend&) = delete;

  void reset() override;
  const RobotState& read_state() noexcept override;
  void write_target(std::span<const float> target) noexcept override;
  void damp() noexcept override;
  void wait() noexcept override {}
  bool healthy(double state_absent_ms) const noexcept override;

  bool state_ready() const noexcept;
  void begin_initialization(double duration_seconds);
  void arm_control();
  void force_damp() noexcept;
  UnitreeMode mode() const noexcept { return mode_.load(); }
  UnitreeWriterStats writer_stats() const noexcept;

 private:
  struct Impl;
  struct StateSlot;
  struct StateSnapshot;
  struct CommandSlot;
  struct CommandSnapshot;

  void low_state_handler(const void* message) noexcept;
  void writer_loop() noexcept;
  bool snapshot_state(StateSnapshot& destination) const noexcept;
  bool snapshot_command(CommandSnapshot& destination) const noexcept;

  std::unique_ptr<Impl> impl_;
  std::array<std::size_t, kJointCount> isaac_to_sdk_{};
  std::array<float, kJointCount> default_joint_position_{};
  std::array<float, kJointCount> stiffness_{};
  std::array<float, kJointCount> damping_{};
  std::array<float, kJointCount> joint_lower_{};
  std::array<float, kJointCount> joint_upper_{};
  std::unique_ptr<StateSlot> state_slot_;
  std::unique_ptr<CommandSlot> command_slot_;
  RobotState state_cache_{};
  std::array<float, kJointCount> init_start_position_{};
  std::atomic<UnitreeMode> mode_{UnitreeMode::kDisabled};
  std::atomic<bool> write_gate_open_{false};
  std::atomic<bool> stop_requested_{false};
  bool writes_enabled_ = false;
  int writer_cpu_ = -1;
  int writer_fifo_priority_ = 0;
  bool lock_memory_ = false;
  bool require_realtime_ = false;
  double state_absent_ms_ = 500.0;
  double command_stale_ms_ = 500.0;
  std::atomic<bool> realtime_configured_{false};
  std::atomic<bool> hardware_fault_latched_{false};
  double init_duration_seconds_ = 3.0;
  std::uint64_t init_start_ns_ = 0;
  std::thread writer_thread_;

  std::atomic<std::uint64_t> writer_ticks_{0};
  std::atomic<std::uint64_t> publishes_{0};
  std::atomic<std::uint64_t> publish_failures_{0};
  std::atomic<std::uint64_t> crc_errors_{0};
  std::atomic<std::uint64_t> hardware_faults_{0};
  std::atomic<std::uint64_t> watchdog_faults_{0};
  std::atomic<std::uint64_t> wake_late_ns_max_{0};
  std::atomic<std::uint64_t> deadline_misses_{0};
};

}  // namespace ec_native
