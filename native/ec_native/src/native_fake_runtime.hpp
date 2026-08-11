#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "native_tracker_core.hpp"
#include "native_backend.hpp"
#include "onnx_engine.hpp"
#include "shm_command_slot.hpp"

namespace ec_native {

constexpr std::size_t kPlannerFrameWidth = 93;
constexpr std::size_t kPlannerHistoryFrames = 10;
constexpr std::size_t kPlannerHistoryWidth =
    kPlannerFrameWidth * kPlannerHistoryFrames;

enum class RuntimeMode : std::uint32_t {
  kStopped = 0,
  kWait = 1,
  kControl = 2,
  kDamp = 3,
};

enum class RuntimeFault : std::uint32_t {
  kNone = 0,
  kCommandAbsent = 1,
  kCommandStale = 2,
  kCommandContract = 3,
  kTracker = 4,
  kRealtimeSetup = 5,
  kStateAbsent = 6,
};

struct NativeRuntimeStats {
  std::uint64_t ticks = 0;
  std::uint64_t control_ticks = 0;
  std::uint64_t wait_ticks = 0;
  std::uint64_t damp_ticks = 0;
  std::uint64_t deadline_misses = 0;
  std::uint64_t planner_requests = 0;
  std::uint64_t planner_responses = 0;
  std::uint64_t response_overruns = 0;
  std::uint64_t scheduler_deadlines_missed = 0;
  std::uint64_t tick_ns_max = 0;
  std::uint64_t wake_late_ns_max = 0;
  std::uint64_t last_chunk_offset_steps = 0;
  std::uint64_t backend_steps = 0;
  std::uint64_t backend_wake_late_ns_max = 0;
  std::uint64_t backend_deadline_misses = 0;
  RuntimeMode mode = RuntimeMode::kStopped;
  RuntimeFault fault = RuntimeFault::kNone;
  bool realtime_configured = false;
  bool backend_realtime_configured = false;
};

struct NativePlannerConfig {
  std::size_t hold_steps = 10;
  std::size_t lead_ticks = 4;
  std::size_t root_qpos_width = 38;
  std::size_t window_frames = 10;
  std::size_t z_dim = 256;
  bool sin_cos_phase = true;
  std::uint32_t direct_tag = 1;
  bool oracle_reference = false;
};

struct NativeSchedulerConfig {
  std::size_t control_hz = 50;
  float lag_alpha = 1.0F;
  std::size_t command_absent_ticks = 100;
  double command_stale_ms = 500.0;
  double state_absent_ms = 500.0;
  int cpu = -1;
  int fifo_priority = 0;
  bool lock_memory = false;
  bool require_realtime = false;
};

// A complete native control loop for the deterministic fake robot backend.
// The thread reads planner responses, assembles causal history, runs the
// optional encoder and policy, and applies targets without a Python callback.
class NativeFakeRuntime {
 public:
  NativeFakeRuntime(
      NativeTrackerCore& tracker, const std::string& response_slot_name,
      const std::string& request_slot_name, bool create_slots,
      const NativeSchedulerConfig& scheduler,
      const NativePlannerConfig& planner,
      const std::string& encoder_path = {},
      const std::string& encoder_input_name = {},
      const std::string& encoder_output_name = {},
      std::size_t encoder_input_width = 0,
      std::size_t encoder_output_width = 0);
  NativeFakeRuntime(
      NativeTrackerCore& tracker, const std::string& response_slot_name,
      const std::string& request_slot_name, bool create_slots,
      const NativeSchedulerConfig& scheduler,
      const NativePlannerConfig& planner,
      std::unique_ptr<NativeRobotBackend> backend,
      const std::string& encoder_path = {},
      const std::string& encoder_input_name = {},
      const std::string& encoder_output_name = {},
      std::size_t encoder_input_width = 0,
      std::size_t encoder_output_width = 0);
  ~NativeFakeRuntime();

  NativeFakeRuntime(const NativeFakeRuntime&) = delete;
  NativeFakeRuntime& operator=(const NativeFakeRuntime&) = delete;

  void start(std::size_t max_ticks, bool paced = true);
  void stop() noexcept;
  void wait();
  bool running() const noexcept { return running_.load(); }

  void set_initial_pose(std::span<const float> pose);
  NativeRuntimeStats stats() const noexcept;
  std::vector<std::uint64_t> tick_durations_ns() const;
  std::vector<float> base_heights() const;
  std::vector<float> reference_joint_mae() const;
  std::vector<std::int32_t> reference_frames() const;
  std::vector<float> joint_position_log() const;
  std::vector<float> anchor_pose_log() const;
  RobotState state() const;
  double backend_time() const;
  double base_height() const;
  double min_base_height() const;

 private:
  static constexpr std::uint32_t kExplicitTag = 0;
  static constexpr std::uint32_t kLatentTag = 1;
  static constexpr std::uint32_t kChunkTag = 2;
  static constexpr std::uint32_t kRawReferenceTag = 3;
  static constexpr std::uint32_t kPlannerRequestTag = 10;
  static constexpr std::uint32_t kOracleRequestTag = 11;
  static constexpr std::size_t kRawReferenceWidth = kJointCount + 3 + 4;
  static constexpr std::size_t kReferenceHeaderWidth = 3;

  void run(std::size_t max_ticks, bool paced) noexcept;
  bool configure_thread() noexcept;
  void one_tick() noexcept;
  void append_planner_frame() noexcept;
  void publish_planner_request() noexcept;
  void read_response() noexcept;
  bool accept_direct_command(std::uint32_t length) noexcept;
  bool accept_chunk(std::uint32_t length) noexcept;
  bool accept_reference_chunk(std::uint32_t length) noexcept;
  void record_reference_metrics() noexcept;
  void transition_to_damp(RuntimeFault fault) noexcept;
  NativeTrackerCore& tracker_;
  std::unique_ptr<ShmSlot> response_slot_;
  std::unique_ptr<ShmSlot> request_slot_;
  NativeSchedulerConfig scheduler_;
  NativePlannerConfig planner_;
  std::unique_ptr<OnnxEngine> encoder_;
  std::unique_ptr<NativeRobotBackend> backend_;

  RobotState robot_state_{};
  std::array<float, kMaxValues> response_buffer_{};
  std::array<float, kMaxCommand> command_{};
  std::array<float, kMaxValues> pending_chunk_{};
  std::array<float, kMaxValues> active_reference_chunk_{};
  std::array<float, kMaxValues> encoder_window_{};
  std::uint32_t pending_chunk_length_ = 0;
  std::uint64_t pending_chunk_sequence_ = 0;
  double pending_chunk_recv_stamp_ = 0.0;
  std::array<float, kPlannerHistoryWidth> planner_history_{};
  bool planner_history_initialized_ = false;
  bool command_available_ = false;
  std::size_t steps_remaining_ = 0;
  std::size_t absent_ticks_ = 0;
  std::uint64_t last_response_sequence_ = 0;
  std::uint64_t consumed_response_sequence_ = 0;
  std::uint64_t request_sequence_ = 0;
  std::uint64_t outstanding_request_sequence_ = 0;
  std::uint64_t outstanding_request_tick_ = 0;
  std::uint64_t outstanding_reference_tick_ = 0;
  std::uint64_t pending_chunk_request_tick_ = 0;
  std::uint64_t pending_reference_tick_ = 0;
  std::uint64_t active_reference_tick_ = 0;
  std::uint32_t active_reference_length_ = 0;
  std::uint32_t active_reference_valid_frames_ = 0;
  std::uint64_t loop_tick_ = 0;
  std::uint64_t episode_generation_ = 0;
  bool pending_chunk_has_request_timing_ = false;
  double command_recv_stamp_ = 0.0;

  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::thread thread_;
  std::vector<std::uint64_t> tick_durations_;
  std::vector<float> base_heights_;
  std::vector<float> reference_joint_mae_;
  std::vector<std::int32_t> reference_frames_;
  std::vector<float> joint_position_log_;  // ticks x kJointCount
  std::vector<float> anchor_pose_log_;     // ticks x 7 (pos, quat XYZW)

  std::atomic<std::uint64_t> ticks_{0};
  std::atomic<std::uint64_t> control_ticks_{0};
  std::atomic<std::uint64_t> wait_ticks_{0};
  std::atomic<std::uint64_t> damp_ticks_{0};
  std::atomic<std::uint64_t> deadline_misses_{0};
  std::atomic<std::uint64_t> planner_requests_{0};
  std::atomic<std::uint64_t> planner_responses_{0};
  std::atomic<std::uint64_t> response_overruns_{0};
  std::atomic<std::uint64_t> scheduler_deadlines_missed_{0};
  std::atomic<std::uint64_t> tick_ns_max_{0};
  std::atomic<std::uint64_t> wake_late_ns_max_{0};
  std::atomic<std::uint64_t> last_chunk_offset_steps_{0};
  std::atomic<RuntimeMode> mode_{RuntimeMode::kStopped};
  std::atomic<RuntimeFault> fault_{RuntimeFault::kNone};
  std::atomic<bool> realtime_configured_{false};
};

}  // namespace ec_native
