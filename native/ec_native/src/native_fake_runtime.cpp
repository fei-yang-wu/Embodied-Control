#include "native_fake_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>

namespace ec_native {
namespace {

std::uint64_t monotonic_ns() noexcept {
  timespec now{};
  clock_gettime(CLOCK_MONOTONIC, &now);
  return static_cast<std::uint64_t>(now.tv_sec) * 1000000000ULL +
         static_cast<std::uint64_t>(now.tv_nsec);
}

timespec to_timespec(std::uint64_t nanoseconds) noexcept {
  return timespec{
      static_cast<time_t>(nanoseconds / 1000000000ULL),
      static_cast<long>(nanoseconds % 1000000000ULL),
  };
}

void update_max(std::atomic<std::uint64_t>& destination,
                std::uint64_t value) noexcept {
  std::uint64_t previous = destination.load(std::memory_order_relaxed);
  while (previous < value &&
         !destination.compare_exchange_weak(previous, value,
                                            std::memory_order_relaxed)) {
  }
}

}  // namespace

NativeFakeRuntime::NativeFakeRuntime(
    NativeTrackerCore& tracker, const std::string& response_slot_name,
    const std::string& request_slot_name, bool create_slots,
    const NativeSchedulerConfig& scheduler,
    const NativePlannerConfig& planner, const std::string& encoder_path,
    const std::string& encoder_input_name,
    const std::string& encoder_output_name, std::size_t encoder_input_width,
    std::size_t encoder_output_width)
    : NativeFakeRuntime(
          tracker, response_slot_name, request_slot_name, create_slots,
          scheduler, planner,
          std::make_unique<NativeFakeBackend>(
              tracker.default_joint_position(), scheduler.control_hz,
              scheduler.lag_alpha),
          encoder_path, encoder_input_name, encoder_output_name,
          encoder_input_width, encoder_output_width) {}

NativeFakeRuntime::NativeFakeRuntime(
    NativeTrackerCore& tracker, const std::string& response_slot_name,
    const std::string& request_slot_name, bool create_slots,
    const NativeSchedulerConfig& scheduler,
    const NativePlannerConfig& planner,
    std::unique_ptr<NativeRobotBackend> backend,
    const std::string& encoder_path,
    const std::string& encoder_input_name,
    const std::string& encoder_output_name, std::size_t encoder_input_width,
    std::size_t encoder_output_width)
    : tracker_(tracker),
      response_slot_(
          std::make_unique<ShmSlot>(response_slot_name, create_slots)),
      scheduler_(scheduler),
      planner_(planner),
      backend_(std::move(backend)) {
  if (!request_slot_name.empty()) {
    request_slot_ =
        std::make_unique<ShmSlot>(request_slot_name, create_slots);
  }
  if (scheduler_.control_hz == 0 || scheduler_.lag_alpha <= 0.0F ||
      scheduler_.lag_alpha > 1.0F || scheduler_.command_absent_ticks == 0 ||
      scheduler_.command_stale_ms <= 0.0 || scheduler_.state_absent_ms <= 0.0 ||
      scheduler_.cpu < -1 || scheduler_.cpu >= CPU_SETSIZE ||
      scheduler_.fifo_priority < 0 ||
      scheduler_.fifo_priority > sched_get_priority_max(SCHED_FIFO)) {
    throw std::runtime_error("invalid native scheduler configuration");
  }
  if (!backend_) {
    throw std::runtime_error("native runtime requires a robot backend");
  }
  if (planner_.plan_slots == 0) {
    throw std::runtime_error("invalid native planner configuration");
  }
  // The request fires `lead_ticks` before the whole plan is consumed, so the
  // lead may span several holds once a reply carries more than one slot. At
  // plan_slots = 1 this is the historical `lead_ticks < hold_steps` rule.
  if (planner_.hold_steps == 0 ||
      planner_.lead_ticks >= planner_.hold_steps * planner_.plan_slots ||
      planner_.encoder_frame_width == 0 || planner_.window_frames == 0 ||
      planner_.encoder_frame_stride == 0 ||
      planner_.z_dim == 0 || planner_.direct_tag > kLatentTag) {
    throw std::runtime_error("invalid native planner configuration");
  }
  if (planner_.latent_plan) {
    if (planner_.oracle_reference) {
      throw std::runtime_error(
          "the latent plan tag and the oracle reference path are exclusive");
    }
    if (planner_.plan_slots * planner_.z_dim > kMaxValues) {
      throw std::runtime_error("latent plan exceeds the shared-memory packet");
    }
    const std::size_t phase_width = planner_.sin_cos_phase ? 2 : 0;
    if (tracker_.command_width() != planner_.z_dim + phase_width) {
      throw std::runtime_error(
          "tracker command width does not match the latent plan plus phase");
    }
  } else if (planner_.plan_slots != 1) {
    throw std::runtime_error(
        "multi-slot plans need the latent plan response tag");
  }

  if (!encoder_path.empty()) {
    raw_reference_width_ =
        planner_.reference_encoder_layout !=
                NativePlannerConfig::ReferenceEncoderLayout::kJointQposQvelAnchorOri
            ? kJointCount + 3 + 4
            : kJointCount * 2 + 4;
    if (encoder_input_width == 0 || encoder_output_width != planner_.z_dim ||
        encoder_input_width !=
            planner_.encoder_frame_width * planner_.window_frames) {
      throw std::runtime_error("encoder dimensions do not match planner contract");
    }
    encoder_ = std::make_unique<OnnxEngine>(
        encoder_path, encoder_input_name, encoder_output_name,
        encoder_input_width, encoder_output_width, 1);
    const std::size_t phase_width = planner_.sin_cos_phase ? 2 : 0;
    if (tracker_.command_width() != planner_.z_dim + phase_width) {
      throw std::runtime_error(
          "tracker command width does not match encoder plus phase");
    }
  }
  if (planner_.oracle_reference &&
      (!request_slot_ || !encoder_ || planner_.window_frames == 0 ||
       kReferenceHeaderWidth +
               ((planner_.window_frames - 1) *
                    planner_.encoder_frame_stride +
                planner_.hold_steps) *
                   raw_reference_width_ >
           kMaxValues)) {
    throw std::runtime_error(
        "native oracle reference block exceeds the encoder contract");
  }

  robot_state_ = backend_->read_state();
}

NativeFakeRuntime::~NativeFakeRuntime() {
  stop();
  wait();
}

void NativeFakeRuntime::start(std::size_t max_ticks, bool paced) {
  if (max_ticks == 0) {
    throw std::runtime_error("max_ticks must be positive");
  }
  if (running_.load() || thread_.joinable()) {
    throw std::runtime_error("native runtime is already started");
  }
  if (!paced && backend_->requires_pacing()) {
    throw std::runtime_error(
        "this native backend requires paced wall-clock execution");
  }
  tick_durations_.assign(max_ticks, 0);
  base_heights_.assign(max_ticks, std::numeric_limits<float>::quiet_NaN());
  reference_joint_mae_.assign(
      max_ticks, std::numeric_limits<float>::quiet_NaN());
  reference_frames_.assign(max_ticks, -1);
  joint_position_log_.assign(max_ticks * kJointCount,
                             std::numeric_limits<float>::quiet_NaN());
  anchor_pose_log_.assign(max_ticks * 7,
                          std::numeric_limits<float>::quiet_NaN());
  backend_->reset();
  robot_state_ = backend_->read_state();
  planner_history_initialized_ = false;
  command_available_ = false;
  steps_remaining_ = 0;
  absent_ticks_ = 0;
  pending_chunk_length_ = 0;
  pending_chunk_sequence_ = 0;
  pending_chunk_recv_stamp_ = 0.0;
  pending_chunk_request_tick_ = 0;
  pending_reference_tick_ = 0;
  active_reference_tick_ = 0;
  active_reference_length_ = 0;
  active_reference_valid_frames_ = 0;
  last_encoder_tick_ = 0;
  encoder_ran_ = false;
  pending_chunk_has_request_timing_ = false;
  active_plan_slots_ = 0;
  plan_cursor_ = 0;
  plan_accept_tick_ = 0;
  consumed_response_sequence_ = last_response_sequence_;
  outstanding_request_sequence_ = 0;
  outstanding_request_tick_ = 0;
  outstanding_reference_tick_ = 0;
  loop_tick_ = 0;
  reference_tick_ = 0;
  ++episode_generation_;
  command_recv_stamp_ = 0.0;
  command_.fill(0.0F);
  tracker_.reset();
  if (encoder_) {
    encoder_->warmup(8);
  }
  tracker_.warmup(8);
  stop_requested_.store(false);
  ticks_.store(0);
  control_ticks_.store(0);
  wait_ticks_.store(0);
  damp_ticks_.store(0);
  deadline_misses_.store(0);
  planner_requests_.store(0);
  planner_responses_.store(0);
  encoder_inferences_.store(0);
  response_overruns_.store(0);
  stale_responses_.store(0);
  plan_slot_advances_.store(0);
  plan_late_starts_.store(0);
  scheduler_deadlines_missed_.store(0);
  tick_ns_max_.store(0);
  wake_late_ns_max_.store(0);
  last_chunk_offset_steps_.store(0);
  fault_.store(RuntimeFault::kNone);
  realtime_configured_.store(false);
  mode_.store(RuntimeMode::kWait);
  if (planner_.oracle_reference) {
    publish_planner_request();
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (outstanding_request_sequence_ != 0 &&
           std::chrono::steady_clock::now() < deadline) {
      read_response();
      if (outstanding_request_sequence_ != 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    }
    if (outstanding_request_sequence_ != 0 ||
        fault_.load() != RuntimeFault::kNone) {
      throw std::runtime_error(
          "native oracle worker did not provide a valid startup response");
    }
  }
  backend_->start(paced);
  running_.store(true);
  try {
    thread_ = std::thread(&NativeFakeRuntime::run, this, max_ticks, paced);
  } catch (...) {
    running_.store(false);
    backend_->stop();
    backend_->wait_for_stop();
    throw;
  }
}

void NativeFakeRuntime::stop() noexcept {
  stop_requested_.store(true);
  backend_->stop();
}

void NativeFakeRuntime::wait() {
  if (thread_.joinable()) {
    thread_.join();
  }
  backend_->stop();
  backend_->wait_for_stop();
}

bool NativeFakeRuntime::configure_thread() noexcept {
  bool configured = true;
  if (scheduler_.lock_memory && mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    configured = false;
  }
  if (scheduler_.cpu >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(scheduler_.cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
      configured = false;
    }
  }
  if (scheduler_.fifo_priority > 0) {
    sched_param parameters{};
    parameters.sched_priority = scheduler_.fifo_priority;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &parameters) != 0) {
      configured = false;
    }
  }
  realtime_configured_.store(configured);
  if (!configured && scheduler_.require_realtime) {
    transition_to_damp(RuntimeFault::kRealtimeSetup);
    return false;
  }
  return true;
}

void NativeFakeRuntime::run(std::size_t max_ticks, bool paced) noexcept {
  configure_thread();
  const std::uint64_t period_ns =
      1000000000ULL / static_cast<std::uint64_t>(scheduler_.control_hz);
  std::uint64_t scheduled_ns = monotonic_ns();

  for (std::size_t tick = 0;
       tick < max_ticks && !stop_requested_.load(std::memory_order_relaxed);
       ++tick) {
    if (paced && tick > 0) {
      scheduled_ns += period_ns;
      const timespec target = to_timespec(scheduled_ns);
      while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &target,
                             nullptr) == EINTR) {
      }
      const std::uint64_t woke_ns = monotonic_ns();
      if (woke_ns > scheduled_ns) {
        update_max(wake_late_ns_max_, woke_ns - scheduled_ns);
      }
    }

    const std::uint64_t started_ns = monotonic_ns();
    loop_tick_ = tick;
    if (tick > 0 && !reference_paused_.load(std::memory_order_relaxed)) {
      ++reference_tick_;
    }
    one_tick();
    const std::uint64_t finished_ns = monotonic_ns();
    const std::uint64_t elapsed_ns = finished_ns - started_ns;
    tick_durations_[tick] = elapsed_ns;
    update_max(tick_ns_max_, elapsed_ns);
    ticks_.store(tick + 1, std::memory_order_release);
    const bool missed = paced ? finished_ns > scheduled_ns + period_ns
                              : elapsed_ns > period_ns;
    if (missed) {
      scheduler_deadlines_missed_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  if (paced && !stop_requested_.load(std::memory_order_relaxed)) {
    const timespec terminal = to_timespec(scheduled_ns + period_ns);
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &terminal,
                           nullptr) == EINTR) {
    }
  }
  running_.store(false);
  backend_->stop();
  if (mode_.load() != RuntimeMode::kDamp) {
    mode_.store(RuntimeMode::kStopped);
  }
}

void NativeFakeRuntime::append_planner_frame() noexcept {
  std::array<float, kPlannerFrameWidth> frame{};
  std::size_t offset = 0;
  for (std::size_t index = 0; index < kJointCount; ++index) {
    frame[offset++] = robot_state_.joint_position[index] -
                      tracker_.default_joint_position()[index];
  }
  for (float value : robot_state_.joint_velocity) {
    frame[offset++] = value;
  }
  for (float value : robot_state_.base_angular_velocity) {
    frame[offset++] = value;
  }
  for (float value : robot_state_.projected_gravity) {
    frame[offset++] = value;
  }
  for (float value : tracker_.last_action()) {
    frame[offset++] = value;
  }
  if (!planner_history_initialized_) {
    for (std::size_t slot = 0; slot < kPlannerHistoryFrames; ++slot) {
      std::copy(frame.begin(), frame.end(),
                planner_history_.begin() + slot * kPlannerFrameWidth);
    }
    planner_history_initialized_ = true;
    return;
  }
  std::move(planner_history_.begin() + kPlannerFrameWidth,
            planner_history_.end(), planner_history_.begin());
  std::copy(frame.begin(), frame.end(),
            planner_history_.end() - kPlannerFrameWidth);
}

void NativeFakeRuntime::publish_planner_request() noexcept {
  if (!request_slot_ || outstanding_request_sequence_ != 0) {
    return;
  }
  try {
    ++request_sequence_;
    if (planner_.oracle_reference) {
      const bool paused = reference_paused_.load(std::memory_order_relaxed);
      outstanding_reference_tick_ = command_available_ && !paused
                                        ? reference_tick_ + steps_remaining_
                                        : reference_tick_;
      const std::array<float, 2> request = {
          static_cast<float>(episode_generation_),
          static_cast<float>(outstanding_reference_tick_),
      };
      request_slot_->publish(request_sequence_, kOracleRequestTag,
                             request.data(), request.size(), monotonic_now());
    } else {
      request_slot_->publish(request_sequence_, kPlannerRequestTag,
                             planner_history_.data(), kPlannerHistoryWidth,
                             monotonic_now());
    }
    outstanding_request_sequence_ = request_sequence_;
    outstanding_request_tick_ = loop_tick_;
    planner_requests_.fetch_add(1, std::memory_order_relaxed);
  } catch (...) {
    transition_to_damp(RuntimeFault::kCommandContract);
  }
}

void NativeFakeRuntime::read_response() noexcept {
  std::uint64_t sequence = 0;
  std::uint32_t tag = 0;
  std::uint32_t length = 0;
  double recv_stamp = 0.0;
  double sender_stamp = 0.0;
  try {
    if (!response_slot_->snapshot(sequence, tag, length, recv_stamp,
                                  sender_stamp, response_buffer_.data(),
                                  response_buffer_.size(),
                                  last_response_sequence_)) {
      return;
    }
  } catch (...) {
    transition_to_damp(RuntimeFault::kCommandContract);
    return;
  }
  static_cast<void>(sender_stamp);
  last_response_sequence_ = sequence;
  planner_responses_.fetch_add(1, std::memory_order_relaxed);
  if (request_slot_ && sequence < outstanding_request_sequence_) {
    // A reply to a request from before the last stop(): the planner stays up
    // across episodes and answers whatever was in flight when the control
    // thread went away. It is stale, not a broken contract.
    stale_responses_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  if (request_slot_ && sequence != outstanding_request_sequence_) {
    transition_to_damp(RuntimeFault::kCommandContract);
    return;
  }

  if (tag == kLatentPlanTag && planner_.latent_plan) {
    if (pending_chunk_sequence_ > consumed_response_sequence_) {
      response_overruns_.fetch_add(1, std::memory_order_relaxed);
    }
    std::copy_n(response_buffer_.begin(), length, pending_chunk_.begin());
    pending_chunk_length_ = length;
    pending_chunk_sequence_ = sequence;
    pending_chunk_recv_stamp_ = recv_stamp;
    pending_chunk_request_tick_ = outstanding_request_tick_;
    pending_chunk_has_request_timing_ = request_slot_ != nullptr;
    outstanding_request_sequence_ = 0;
    return;
  }
  if (((tag == kChunkTag && !planner_.oracle_reference) ||
       (tag == kRawReferenceTag && planner_.oracle_reference)) &&
      encoder_) {
    if (tag == kRawReferenceTag) {
      if (length < kReferenceHeaderWidth ||
          response_buffer_[0] != static_cast<float>(episode_generation_) ||
          response_buffer_[1] !=
              static_cast<float>(outstanding_reference_tick_)) {
        transition_to_damp(RuntimeFault::kCommandContract);
        return;
      }
      pending_reference_tick_ = outstanding_reference_tick_;
    }
    if (pending_chunk_sequence_ > consumed_response_sequence_) {
      response_overruns_.fetch_add(1, std::memory_order_relaxed);
    }
    std::copy_n(response_buffer_.begin(), length, pending_chunk_.begin());
    pending_chunk_length_ = length;
    pending_chunk_sequence_ = sequence;
    pending_chunk_recv_stamp_ = recv_stamp;
    pending_chunk_request_tick_ = outstanding_request_tick_;
    pending_chunk_has_request_timing_ = request_slot_ != nullptr;
    outstanding_request_sequence_ = 0;
    return;
  }
  if (!planner_.oracle_reference && tag == planner_.direct_tag) {
    if (!accept_direct_command(length)) {
      transition_to_damp(RuntimeFault::kCommandContract);
      return;
    }
    consumed_response_sequence_ = sequence;
    command_recv_stamp_ = recv_stamp;
    outstanding_request_sequence_ = 0;
    return;
  }
  transition_to_damp(RuntimeFault::kCommandContract);
}

bool NativeFakeRuntime::accept_direct_command(std::uint32_t length) noexcept {
  if (length != tracker_.command_width() || length > command_.size()) {
    return false;
  }
  if (!std::all_of(response_buffer_.begin(), response_buffer_.begin() + length,
                   [](float value) { return std::isfinite(value); })) {
    return false;
  }
  std::copy_n(response_buffer_.begin(), length, command_.begin());
  command_available_ = true;
  steps_remaining_ = planner_.hold_steps;
  absent_ticks_ = 0;
  // A pushed command supersedes any plan in hand; never walk a stale plan
  // after it expires.
  active_plan_slots_ = 0;
  plan_cursor_ = 0;
  mode_.store(RuntimeMode::kControl);
  return true;
}

std::size_t NativeFakeRuntime::plan_ticks_remaining() const noexcept {
  if (active_plan_slots_ == 0) {
    return steps_remaining_;
  }
  return steps_remaining_ +
         (active_plan_slots_ - 1 - plan_cursor_) * planner_.hold_steps;
}

void NativeFakeRuntime::load_plan_slot(std::size_t slot) noexcept {
  const float* source = active_plan_.data() + slot * planner_.z_dim;
  std::copy_n(source, planner_.z_dim, command_.begin());
  plan_cursor_ = slot;
  command_available_ = true;
  absent_ticks_ = 0;
  mode_.store(RuntimeMode::kControl);
}

// Serve the next command from the plan already in hand. No planner call, no
// transport: this is what makes one head inference cover plan_slots holds.
bool NativeFakeRuntime::advance_plan_slot() noexcept {
  if (active_plan_slots_ == 0 || plan_cursor_ + 1 >= active_plan_slots_) {
    return false;
  }
  load_plan_slot(plan_cursor_ + 1);
  steps_remaining_ = planner_.hold_steps;
  plan_slot_advances_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool NativeFakeRuntime::accept_latent_plan(std::uint32_t length) noexcept {
  if (!planner_.latent_plan || planner_.z_dim == 0 ||
      length % planner_.z_dim != 0) {
    return false;
  }
  const std::size_t slots = length / planner_.z_dim;
  if (slots == 0 || slots > planner_.plan_slots) {
    return false;
  }
  if (!std::all_of(pending_chunk_.begin(), pending_chunk_.begin() + length,
                   [](float value) { return std::isfinite(value); })) {
    return false;
  }
  // Time-align the plan the way the pull client does: the slots whose time
  // passed while the planner was thinking are never replayed.
  const std::size_t elapsed =
      pending_chunk_has_request_timing_
          ? static_cast<std::size_t>(loop_tick_ - pending_chunk_request_tick_)
          : planner_.lead_ticks;
  std::size_t start_slot = elapsed / planner_.hold_steps;
  std::size_t into_slot = elapsed % planner_.hold_steps;
  if (start_slot >= slots) {
    // The whole plan is already in the past. Fall back to its last slot and
    // record it: a run with plan_late_starts > 0 was not served fresh data.
    start_slot = slots - 1;
    into_slot = 0;
    plan_late_starts_.fetch_add(1, std::memory_order_relaxed);
  }
  std::copy_n(pending_chunk_.begin(), length, active_plan_.begin());
  active_plan_slots_ = slots;
  plan_accept_tick_ = loop_tick_;
  load_plan_slot(start_slot);
  steps_remaining_ = planner_.hold_steps - into_slot;
  consumed_response_sequence_ = pending_chunk_sequence_;
  command_recv_stamp_ = pending_chunk_recv_stamp_;
  pending_chunk_length_ = 0;
  pending_chunk_sequence_ = 0;
  pending_chunk_request_tick_ = 0;
  pending_chunk_has_request_timing_ = false;
  last_chunk_offset_steps_.store(elapsed, std::memory_order_relaxed);
  return true;
}

bool NativeFakeRuntime::accept_chunk(std::uint32_t length) noexcept {
  if (!encoder_) {
    return false;
  }
  const std::size_t input_width = encoder_->input_width();
  const float* input = pending_chunk_.data();
  std::size_t offset_steps = 0;
  if (length == input_width) {
    // A service can publish an already selected encoder window.
  } else {
    offset_steps = pending_chunk_has_request_timing_
                       ? loop_tick_ - pending_chunk_request_tick_
                       : planner_.lead_ticks;
    const std::size_t start = offset_steps * planner_.encoder_frame_width;
    if (length < start + input_width) {
      return false;
    }
    input += start;
  }
  if (!std::all_of(input, input + input_width,
                   [](float value) { return std::isfinite(value); })) {
    return false;
  }
  try {
    const auto z = encoder_->infer(std::span<const float>(input, input_width));
    encoder_inferences_.fetch_add(1, std::memory_order_relaxed);
    if (z.size() != planner_.z_dim) {
      return false;
    }
    std::copy(z.begin(), z.end(), command_.begin());
  } catch (...) {
    return false;
  }
  command_available_ = true;
  steps_remaining_ = planner_.hold_steps;
  absent_ticks_ = 0;
  consumed_response_sequence_ = pending_chunk_sequence_;
  command_recv_stamp_ = pending_chunk_recv_stamp_;
  pending_chunk_length_ = 0;
  pending_chunk_sequence_ = 0;
  pending_chunk_request_tick_ = 0;
  pending_chunk_has_request_timing_ = false;
  last_chunk_offset_steps_.store(offset_steps, std::memory_order_relaxed);
  mode_.store(RuntimeMode::kControl);
  return true;
}

bool NativeFakeRuntime::accept_reference_chunk(std::uint32_t length) noexcept {
  if (!encoder_ || !robot_state_.anchor_pose_valid ||
      length < kReferenceHeaderWidth ||
      response_buffer_.size() < length) {
    return false;
  }
  const float valid_frames_value = pending_chunk_[2];
  if (!std::isfinite(valid_frames_value) || valid_frames_value < 1.0F) {
    return false;
  }
  const std::size_t available_frames =
      (length - kReferenceHeaderWidth) / raw_reference_width_;
  if (length !=
          kReferenceHeaderWidth + available_frames * raw_reference_width_ ||
      available_frames <
          (planner_.window_frames - 1) * planner_.encoder_frame_stride + 1) {
    return false;
  }
  const std::size_t offset_steps =
      reference_tick_ >= pending_reference_tick_
          ? static_cast<std::size_t>(reference_tick_ - pending_reference_tick_)
          : 0;
  if (offset_steps +
          (planner_.window_frames - 1) * planner_.encoder_frame_stride >=
      available_frames) {
    return false;
  }
  std::copy_n(pending_chunk_.begin(), length, active_reference_chunk_.begin());
  active_reference_length_ = length;
  active_reference_tick_ = pending_reference_tick_;
  active_reference_valid_frames_ =
      static_cast<std::uint32_t>(valid_frames_value);
  if (!encode_active_reference(offset_steps)) {
    return false;
  }
  command_available_ = true;
  steps_remaining_ = planner_.hold_steps;
  absent_ticks_ = 0;
  consumed_response_sequence_ = pending_chunk_sequence_;
  command_recv_stamp_ = pending_chunk_recv_stamp_;
  pending_chunk_length_ = 0;
  pending_chunk_sequence_ = 0;
  pending_chunk_request_tick_ = 0;
  pending_reference_tick_ = 0;
  pending_chunk_has_request_timing_ = false;
  last_chunk_offset_steps_.store(offset_steps, std::memory_order_relaxed);
  mode_.store(RuntimeMode::kControl);
  return true;
}

bool NativeFakeRuntime::encode_active_reference(
    std::size_t offset_steps) noexcept {
  if (!encoder_ || !robot_state_.anchor_pose_valid ||
      active_reference_length_ < kReferenceHeaderWidth) {
    return false;
  }
  const std::size_t available_frames =
      (active_reference_length_ - kReferenceHeaderWidth) /
      raw_reference_width_;
  if (offset_steps +
          (planner_.window_frames - 1) * planner_.encoder_frame_stride >=
      available_frames) {
    return false;
  }
  const float* raw = active_reference_chunk_.data() + kReferenceHeaderWidth;
  const std::size_t input_width = encoder_->input_width();
  bool packed = false;
  if (planner_.reference_encoder_layout !=
      NativePlannerConfig::ReferenceEncoderLayout::kJointQposQvelAnchorOri) {
    const std::size_t selected_width = planner_.window_frames *
                                       raw_reference_width_;
    if (selected_width <= encoder_raw_window_.size()) {
      for (std::size_t frame = 0; frame < planner_.window_frames; ++frame) {
        const std::size_t source_frame =
            offset_steps + frame * planner_.encoder_frame_stride;
        std::copy_n(raw + source_frame * raw_reference_width_,
                    raw_reference_width_,
                    encoder_raw_window_.data() +
                        frame * raw_reference_width_);
      }
      packed = input_width ==
                   planner_.window_frames * planner_.encoder_frame_width &&
               reexpress_root_qpos_window(
                   std::span<const float>(encoder_raw_window_.data(),
                                          selected_width),
                   planner_.window_frames, robot_state_.anchor_position_w,
                   robot_state_.anchor_quaternion_w,
                   std::span<float>(encoder_window_.data(), input_width),
                   planner_.reference_encoder_layout ==
                       NativePlannerConfig::ReferenceEncoderLayout::kRootQposHeading);
    }
  } else {
    packed = pack_joint_qpos_qvel_anchor_ori_window(
        std::span<const float>(raw,
                               available_frames * raw_reference_width_),
        available_frames, offset_steps, planner_.window_frames,
        planner_.encoder_frame_stride, robot_state_.anchor_quaternion_w,
        std::span<float>(encoder_window_.data(), input_width));
  }
  if (!packed) {
    return false;
  }
  try {
    const auto z = encoder_->infer(
        std::span<const float>(encoder_window_.data(), input_width));
    encoder_inferences_.fetch_add(1, std::memory_order_relaxed);
    if (z.size() != planner_.z_dim) {
      return false;
    }
    std::copy(z.begin(), z.end(), command_.begin());
  } catch (...) {
    return false;
  }
  last_encoder_tick_ = loop_tick_;
  encoder_ran_ = true;
  return true;
}

void NativeFakeRuntime::record_reference_metrics() noexcept {
  if (loop_tick_ < base_heights_.size()) {
    base_heights_[loop_tick_] = static_cast<float>(backend_->base_height());
    std::copy(robot_state_.joint_position.begin(),
              robot_state_.joint_position.end(),
              joint_position_log_.begin() + loop_tick_ * kJointCount);
    if (robot_state_.anchor_pose_valid) {
      float* pose = anchor_pose_log_.data() + loop_tick_ * 7;
      std::copy(robot_state_.anchor_position_w.begin(),
                robot_state_.anchor_position_w.end(), pose);
      std::copy(robot_state_.anchor_quaternion_w.begin(),
                robot_state_.anchor_quaternion_w.end(), pose + 3);
    }
  }
  if (!planner_.oracle_reference || active_reference_length_ == 0 ||
      reference_tick_ < active_reference_tick_ ||
      loop_tick_ >= reference_joint_mae_.size()) {
    return;
  }
  const std::size_t frame =
      static_cast<std::size_t>(reference_tick_ - active_reference_tick_);
  const std::size_t available_frames =
      (active_reference_length_ - kReferenceHeaderWidth) /
      raw_reference_width_;
  if (frame >= available_frames || frame >= active_reference_valid_frames_) {
    return;
  }
  // `frame` is the slot inside the active window; the absolute reference
  // frame is offset by the window's start frame.
  reference_frames_[loop_tick_] =
      static_cast<std::int32_t>(active_reference_tick_ + frame);
  const float* reference = active_reference_chunk_.data() +
                           kReferenceHeaderWidth +
                           frame * raw_reference_width_;
  float error = 0.0F;
  for (std::size_t joint = 0; joint < kJointCount; ++joint) {
    error += std::abs(robot_state_.joint_position[joint] - reference[joint]);
  }
  reference_joint_mae_[loop_tick_] = error / static_cast<float>(kJointCount);
}

void NativeFakeRuntime::one_tick() noexcept {
  robot_state_ = backend_->read_state();
  if (!backend_->healthy(scheduler_.state_absent_ms)) {
    transition_to_damp(RuntimeFault::kStateAbsent);
    backend_->damp();
    damp_ticks_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  append_planner_frame();
  read_response();
  if (mode_.load() == RuntimeMode::kDamp) {
    backend_->damp();
    damp_ticks_.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  // Hold scheduling runs whenever the controller owns command expiry: the
  // encoder path (one window per hold) and the latent plan path (one reply
  // per plan_slots holds). A pushed direct command has no expiry.
  const bool controller_paced_command = encoder_ != nullptr ||
                                        planner_.latent_plan;
  if (!command_available_) {
    if (pending_chunk_sequence_ > consumed_response_sequence_) {
      const bool accepted =
          planner_.latent_plan
              ? accept_latent_plan(pending_chunk_length_)
              : (planner_.oracle_reference
                     ? accept_reference_chunk(pending_chunk_length_)
                     : accept_chunk(pending_chunk_length_));
      if (!accepted) {
        transition_to_damp(RuntimeFault::kCommandContract);
      }
    } else {
      publish_planner_request();
    }
  } else if (controller_paced_command) {
    // `<=`, not `==`: at hold 1 the countdown steps 1 -> 0 and would skip an
    // equality test, and once a plan is exhausted the remaining count stays
    // pinned at the hold, so an equality test would never ask for another
    // plan again. publish_planner_request() keeps at most one in flight.
    if (plan_ticks_remaining() <= planner_.lead_ticks &&
        pending_chunk_sequence_ <= consumed_response_sequence_) {
      publish_planner_request();
    }
    if (steps_remaining_ == 0) {
      if (advance_plan_slot()) {
        // Served from the plan in hand; the planner is not called.
      } else if (pending_chunk_sequence_ > consumed_response_sequence_) {
        const bool accepted =
            planner_.latent_plan
                ? accept_latent_plan(pending_chunk_length_)
                : (planner_.oracle_reference
                       ? accept_reference_chunk(pending_chunk_length_)
                       : accept_chunk(pending_chunk_length_));
        if (!accepted) {
          transition_to_damp(RuntimeFault::kCommandContract);
        }
      } else {
        deadline_misses_.fetch_add(1, std::memory_order_relaxed);
        steps_remaining_ = planner_.hold_steps;
      }
    }
  }

  if (!command_available_) {
    record_reference_metrics();
    ++absent_ticks_;
    wait_ticks_.fetch_add(1, std::memory_order_relaxed);
    backend_->wait();
    if (absent_ticks_ > scheduler_.command_absent_ticks) {
      transition_to_damp(RuntimeFault::kCommandAbsent);
      backend_->damp();
    }
    return;
  }

  record_reference_metrics();

  if (planner_.oracle_reference &&
      planner_.encoder_trigger ==
          NativePlannerConfig::EncoderTrigger::kEveryControlTick &&
      (!encoder_ran_ || last_encoder_tick_ != loop_tick_)) {
    const std::size_t reference_offset =
        reference_tick_ >= active_reference_tick_
            ? static_cast<std::size_t>(reference_tick_ - active_reference_tick_)
            : 0;
    if (!encode_active_reference(reference_offset)) {
      transition_to_damp(RuntimeFault::kCommandContract);
      backend_->damp();
      damp_ticks_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  }

  double command_age_ms = (monotonic_now() - command_recv_stamp_) * 1000.0;
  if (active_plan_slots_ > 0) {
    // A plan is not stale while it is still inside its own horizon: it was
    // predicted for these ticks. Age counts only the time past the horizon
    // the plan covers, so a planner that stops replying still trips the
    // watchdog `command_stale_ms` after the plan runs out.
    const double tick_ms = 1000.0 / static_cast<double>(scheduler_.control_hz);
    const std::uint64_t elapsed_ticks =
        loop_tick_ >= plan_accept_tick_ ? loop_tick_ - plan_accept_tick_ : 0;
    const double covered_ticks = static_cast<double>(std::min<std::uint64_t>(
        elapsed_ticks, active_plan_slots_ * planner_.hold_steps));
    command_age_ms -= covered_ticks * tick_ms;
  }
  if (command_age_ms > scheduler_.command_stale_ms) {
    transition_to_damp(RuntimeFault::kCommandStale);
    backend_->damp();
    damp_ticks_.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  if (controller_paced_command && planner_.sin_cos_phase) {
    const float phase = static_cast<float>(planner_.hold_steps -
                                           steps_remaining_) /
                        static_cast<float>(planner_.hold_steps);
    command_[planner_.z_dim] =
        std::sin(2.0F * static_cast<float>(M_PI) * phase);
    command_[planner_.z_dim + 1] =
        std::cos(2.0F * static_cast<float>(M_PI) * phase);
  }
  try {
    const auto& result = tracker_.step(
        robot_state_,
        std::span<const float>(command_.data(), tracker_.command_width()));
    const float weight = backend_->blend_weight();
    std::array<float, kJointCount> held{};
    if (weight < 1.0F && backend_->held_target(held)) {
      std::array<float, kJointCount> blended{};
      std::array<float, kJointCount> applied_action{};
      const auto& defaults = tracker_.default_joint_position();
      for (std::size_t index = 0; index < kJointCount; ++index) {
        blended[index] = held[index] * (1.0F - weight) +
                         result.joint_target[index] * weight;
        applied_action[index] = result.action[index] * weight;
      }
      backend_->write_target(blended);
      static_cast<void>(defaults);
      tracker_.set_last_action(applied_action);
    } else {
      backend_->write_target(result.joint_target);
    }
    control_ticks_.fetch_add(1, std::memory_order_relaxed);
    if (controller_paced_command && steps_remaining_ > 0) {
      --steps_remaining_;
    }
  } catch (...) {
    transition_to_damp(RuntimeFault::kTracker);
    backend_->damp();
    damp_ticks_.fetch_add(1, std::memory_order_relaxed);
  }
}

void NativeFakeRuntime::transition_to_damp(RuntimeFault fault) noexcept {
  RuntimeFault expected = RuntimeFault::kNone;
  fault_.compare_exchange_strong(expected, fault);
  mode_.store(RuntimeMode::kDamp);
}

NativeRuntimeStats NativeFakeRuntime::stats() const noexcept {
  const NativeBackendTimingStats backend_stats = backend_->timing_stats();
  return NativeRuntimeStats{
      .ticks = ticks_.load(),
      .control_ticks = control_ticks_.load(),
      .wait_ticks = wait_ticks_.load(),
      .damp_ticks = damp_ticks_.load(),
      .deadline_misses = deadline_misses_.load(),
      .planner_requests = planner_requests_.load(),
      .planner_responses = planner_responses_.load(),
      .encoder_inferences = encoder_inferences_.load(),
      .response_overruns = response_overruns_.load(),

      .stale_responses = stale_responses_.load(),


      .reference_ticks = reference_tick_,



      .command_age_ms = command_recv_stamp_ > 0.0 ? (monotonic_now() - command_recv_stamp_) * 1000.0 : -1.0,
      .plan_slot_advances = plan_slot_advances_.load(),
      .plan_late_starts = plan_late_starts_.load(),
      .scheduler_deadlines_missed = scheduler_deadlines_missed_.load(),
      .tick_ns_max = tick_ns_max_.load(),
      .wake_late_ns_max = wake_late_ns_max_.load(),
      .last_chunk_offset_steps = last_chunk_offset_steps_.load(),
      .backend_steps = backend_stats.steps,
      .backend_wake_late_ns_max = backend_stats.wake_late_ns_max,
      .backend_deadline_misses = backend_stats.deadline_misses,
      .mode = mode_.load(),
      .fault = fault_.load(),
      .realtime_configured = realtime_configured_.load(),
      .backend_realtime_configured = backend_stats.realtime_configured,
  };
}

std::vector<std::uint64_t> NativeFakeRuntime::tick_durations_ns() const {
  const std::size_t count = static_cast<std::size_t>(ticks_.load());
  return {tick_durations_.begin(), tick_durations_.begin() + count};
}

std::vector<float> NativeFakeRuntime::base_heights() const {
  const std::size_t count = static_cast<std::size_t>(ticks_.load());
  return {base_heights_.begin(), base_heights_.begin() + count};
}

std::vector<float> NativeFakeRuntime::reference_joint_mae() const {
  const std::size_t count = static_cast<std::size_t>(ticks_.load());
  return {reference_joint_mae_.begin(), reference_joint_mae_.begin() + count};
}

void NativeFakeRuntime::set_initial_pose(std::span<const float> pose) {
  backend_->set_initial_pose(pose);
}

std::vector<std::int32_t> NativeFakeRuntime::reference_frames() const {
  const std::size_t count = static_cast<std::size_t>(ticks_.load());
  return {reference_frames_.begin(), reference_frames_.begin() + count};
}

std::vector<float> NativeFakeRuntime::joint_position_log() const {
  const std::size_t count = static_cast<std::size_t>(ticks_.load());
  return {joint_position_log_.begin(),
          joint_position_log_.begin() + count * kJointCount};
}

std::vector<float> NativeFakeRuntime::anchor_pose_log() const {
  const std::size_t count = static_cast<std::size_t>(ticks_.load());
  return {anchor_pose_log_.begin(), anchor_pose_log_.begin() + count * 7};
}

RobotState NativeFakeRuntime::state() const {
  if (running_.load(std::memory_order_acquire)) {
    throw std::runtime_error("native state is available after the loop stops");
  }
  return robot_state_;
}

double NativeFakeRuntime::backend_time() const {
  if (running_.load(std::memory_order_acquire)) {
    throw std::runtime_error(
        "native backend time is available after the loop stops");
  }
  return backend_->time();
}

double NativeFakeRuntime::base_height() const {
  if (running_.load(std::memory_order_acquire)) {
    throw std::runtime_error(
        "native base height is available after the loop stops");
  }
  return backend_->base_height();
}

double NativeFakeRuntime::min_base_height() const {
  if (running_.load(std::memory_order_acquire)) {
    throw std::runtime_error(
        "native minimum base height is available after the loop stops");
  }
  return backend_->min_base_height();
}

}  // namespace ec_native
