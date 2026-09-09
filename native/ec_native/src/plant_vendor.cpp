#include "plant_vendor.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

#include <unitree/common/json/jsonize.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_api.hpp>
#include <unitree/robot/client/client.hpp>
#include <unitree/robot/g1/loco/g1_loco_api.hpp>
#include <unitree/robot/go2/public/jsonize_type.hpp>
#include <unitree/robot/server/server.hpp>

#include "dds_channel.hpp"

namespace ec_native {
namespace {

constexpr const char* kPlantServiceName = "ec_plant";
constexpr const char* kPlantApiVersion = "1.0.0.0";
// Above the SDK's reserved low ids: the Client base uses the single digits
// for its own handshake, and a plant that answered id 2 with "lower" dropped
// the robot the moment a client connected.
constexpr std::int32_t kPlantApiHoist = 9001;
constexpr std::int32_t kPlantApiLower = 9002;
constexpr std::int32_t kPlantApiStatus = 9003;
constexpr std::int32_t kPlantApiSlack = 9004;
constexpr std::int32_t kPlantApiReset = 9005;

// Nonzero statuses in the vendor's own numbering range, so a refused call
// reads as "the robot said no", never as a transport error.
constexpr std::int32_t kStatusServiceInactive = 7402;
constexpr std::int32_t kStatusRefused = 7403;
constexpr std::int32_t kStatusUnknownService = 7404;

struct SharedState {
  std::atomic<bool> owned{true};
  std::atomic<int> fsm_id{1};
  std::atomic<int> hoist_mode{PlantVendor::kHoistHoisted};
  std::atomic<float> hoist_gain{1.0F};
  std::atomic<std::uint64_t> hoist_generation{1};
  std::atomic<std::uint64_t> reset_generation{0};
  std::atomic<std::uint64_t> reset_applied_generation{0};
  std::string service_name;
};

using unitree::common::FromJsonString;
using unitree::common::ToJsonString;

class SportServer : public unitree::robot::Server {
 public:
  explicit SportServer(std::shared_ptr<SharedState> state)
      : Server(unitree::robot::g1::LOCO_SERVICE_NAME),
        state_(std::move(state)) {}

  void Init() override {
    using namespace unitree::robot::g1;
    SetApiVersion(LOCO_API_VERSION);
    UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(ROBOT_API_ID_LOCO_GET_FSM_ID,
                                             &SportServer::GetFsmId);
    UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(ROBOT_API_ID_LOCO_GET_FSM_MODE,
                                             &SportServer::GetZeroInt);
    UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(
        ROBOT_API_ID_LOCO_GET_BALANCE_MODE, &SportServer::GetZeroInt);
    UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(
        ROBOT_API_ID_LOCO_GET_SWING_HEIGHT, &SportServer::GetZeroFloat);
    UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(
        ROBOT_API_ID_LOCO_GET_STAND_HEIGHT, &SportServer::GetZeroFloat);
    UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(ROBOT_API_ID_LOCO_SET_FSM_ID,
                                             &SportServer::SetFsmId);
    for (const std::int32_t api :
         {ROBOT_API_ID_LOCO_FSM_API, ROBOT_API_ID_LOCO_SET_BALANCE_MODE,
          ROBOT_API_ID_LOCO_SET_SWING_HEIGHT,
          ROBOT_API_ID_LOCO_SET_STAND_HEIGHT, ROBOT_API_ID_LOCO_SET_VELOCITY,
          ROBOT_API_ID_LOCO_SET_ARM_TASK, ROBOT_API_ID_LOCO_SET_SPEED_MODE}) {
      UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(api, &SportServer::Accept);
    }
  }

 private:
  std::int32_t GetFsmId(const std::string&, std::string& data) {
    if (!state_->owned.load()) {
      return kStatusServiceInactive;
    }
    unitree::robot::go2::JsonizeDataInt json;
    json.data = state_->fsm_id.load();
    data = ToJsonString(json);
    return 0;
  }

  std::int32_t GetZeroInt(const std::string&, std::string& data) {
    if (!state_->owned.load()) {
      return kStatusServiceInactive;
    }
    unitree::robot::go2::JsonizeDataInt json;
    data = ToJsonString(json);
    return 0;
  }

  std::int32_t GetZeroFloat(const std::string&, std::string& data) {
    if (!state_->owned.load()) {
      return kStatusServiceInactive;
    }
    unitree::robot::go2::JsonizeDataFloat json;
    data = ToJsonString(json);
    return 0;
  }

  std::int32_t SetFsmId(const std::string& parameter, std::string&) {
    if (!state_->owned.load()) {
      return kStatusServiceInactive;
    }
    unitree::robot::go2::JsonizeDataInt json;
    FromJsonString(parameter, json);
    if (json.data < 0) {
      return kStatusRefused;
    }
    state_->fsm_id.store(json.data);
    return 0;
  }

  std::int32_t Accept(const std::string&, std::string&) {
    return state_->owned.load() ? 0 : kStatusServiceInactive;
  }

  std::shared_ptr<SharedState> state_;
};

class MotionSwitcherServer : public unitree::robot::Server {
 public:
  explicit MotionSwitcherServer(std::shared_ptr<SharedState> state)
      : Server(unitree::robot::b2::MOTION_SWITCHER_SERVICE_NAME),
        state_(std::move(state)) {}

  void Init() override {
    using namespace unitree::robot::b2;
    SetApiVersion(MOTION_SWITCHER_API_VERSION);
    UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(MOTION_SWITCHER_API_ID_CHECK_MODE,
                                             &MotionSwitcherServer::CheckMode);
    UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(
        MOTION_SWITCHER_API_ID_SELECT_MODE, &MotionSwitcherServer::SelectMode);
    UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(
        MOTION_SWITCHER_API_ID_RELEASE_MODE,
        &MotionSwitcherServer::ReleaseMode);
    UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(MOTION_SWITCHER_API_ID_SET_SILENT,
                                             &MotionSwitcherServer::Silent);
    UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(MOTION_SWITCHER_API_ID_GET_SILENT,
                                             &MotionSwitcherServer::Silent);
  }

 private:
  std::int32_t CheckMode(const std::string&, std::string& data) {
    unitree::robot::b2::JsonizeModeName json;
    if (state_->owned.load()) {
      json.name = state_->service_name;
      json.form = "sim";
    }
    data = ToJsonString(json);
    return 0;
  }

  std::int32_t SelectMode(const std::string& parameter, std::string&) {
    unitree::robot::b2::JsonizeModeName json;
    FromJsonString(parameter, json);
    if (json.name != state_->service_name) {
      return kStatusUnknownService;
    }
    // A restarted motion service comes up in damp, which is exactly why the
    // lifecycle hoists before restoring.
    state_->fsm_id.store(1);
    state_->owned.store(true);
    return 0;
  }

  std::int32_t ReleaseMode(const std::string&, std::string&) {
    if (!state_->owned.load()) {
      return 0;
    }
    if (state_->fsm_id.load() != 1) {
      return kStatusRefused;
    }
    state_->owned.store(false);
    return 0;
  }

  std::int32_t Silent(const std::string&, std::string& data) {
    unitree::robot::b2::JsonizeSilent json;
    data = ToJsonString(json);
    return 0;
  }

  std::shared_ptr<SharedState> state_;
};

class PlantServer : public unitree::robot::Server {
 public:
  explicit PlantServer(std::shared_ptr<SharedState> state)
      : Server(kPlantServiceName), state_(std::move(state)) {}

  void Init() override {
    SetApiVersion(kPlantApiVersion);
    UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(kPlantApiHoist,
                                             &PlantServer::Hoist);
    UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(kPlantApiLower,
                                             &PlantServer::Lower);
    UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(kPlantApiStatus,
                                             &PlantServer::Status);
    UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(kPlantApiSlack,
                                             &PlantServer::Slack);
    UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(kPlantApiReset,
                                             &PlantServer::Reset);
  }

 private:
  std::int32_t Hoist(const std::string&, std::string&) {
    state_->hoist_generation.fetch_add(1);
    state_->hoist_mode.store(PlantVendor::kHoistHoisted);
    return 0;
  }

  std::int32_t Lower(const std::string&, std::string&) {
    state_->hoist_mode.store(PlantVendor::kHoistLowered);
    return 0;
  }

  std::int32_t Slack(const std::string&, std::string&) {
    state_->hoist_mode.store(PlantVendor::kHoistSlack);
    return 0;
  }

  std::int32_t Reset(const std::string&, std::string&) {
    state_->fsm_id.store(1);
    state_->owned.store(true);
    state_->hoist_generation.fetch_add(1);
    state_->hoist_mode.store(PlantVendor::kHoistHoisted);
    state_->reset_generation.fetch_add(1);
    return 0;
  }

  std::int32_t Status(const std::string&, std::string& data) {
    const int mode = state_->hoist_mode.load();
    data = "{\"owned\":" + std::string(state_->owned.load() ? "true" : "false") +
           ",\"fsm_id\":" + std::to_string(state_->fsm_id.load()) +
           ",\"hoisted\":" +
           std::string(mode == PlantVendor::kHoistHoisted ? "true" : "false") +
           ",\"hoist_mode\":" + std::to_string(mode) +
           ",\"hoist_gain\":" + std::to_string(state_->hoist_gain.load()) +
           ",\"reset_pending\":" +
           std::string(state_->reset_generation.load() !=
                               state_->reset_applied_generation.load()
                           ? "true"
                           : "false") +
           ",\"service\":\"" + state_->service_name + "\"}";
    return 0;
  }

  std::shared_ptr<SharedState> state_;
};

class PlantClientStub : public unitree::robot::Client {
 public:
  PlantClientStub() : Client(kPlantServiceName, false) {}

  void Init() override {
    SetApiVersion(kPlantApiVersion);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(kPlantApiHoist);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(kPlantApiLower);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(kPlantApiStatus);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(kPlantApiSlack);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(kPlantApiReset);
  }

  std::int32_t Slack() {
    std::string parameter;
    std::string data;
    return Call(kPlantApiSlack, parameter, data);
  }

  std::int32_t Hoist() {
    std::string parameter;
    std::string data;
    return Call(kPlantApiHoist, parameter, data);
  }

  std::int32_t Lower() {
    std::string parameter;
    std::string data;
    return Call(kPlantApiLower, parameter, data);
  }

  std::int32_t Status(std::string& data) {
    std::string parameter;
    return Call(kPlantApiStatus, parameter, data);
  }

  std::int32_t Reset() {
    std::string parameter;
    std::string data;
    return Call(kPlantApiReset, parameter, data);
  }
};

void check(std::int32_t status, const char* verb) {
  if (status != 0) {
    throw std::runtime_error("plant refused " + std::string(verb) +
                             " (status " + std::to_string(status) + ")");
  }
}

}  // namespace

struct PlantVendor::Impl {
  std::shared_ptr<SharedState> state = std::make_shared<SharedState>();
  std::unique_ptr<SportServer> sport;
  std::unique_ptr<MotionSwitcherServer> switcher;
  std::unique_ptr<PlantServer> plant;
};

PlantVendor::PlantVendor(const std::string& service_name, bool vendor_enabled,
                         bool hoisted)
    : impl_(std::make_unique<Impl>()) {
  if (service_name.empty()) {
    throw std::runtime_error("the plant vendor needs a motion service name");
  }
  impl_->state->service_name = service_name;
  impl_->state->owned.store(vendor_enabled);
  impl_->state->hoist_mode.store(hoisted ? kHoistHoisted : kHoistSlack);
  if (vendor_enabled) {
    impl_->sport = std::make_unique<SportServer>(impl_->state);
    impl_->sport->Init();
    impl_->sport->Start();
    impl_->switcher = std::make_unique<MotionSwitcherServer>(impl_->state);
    impl_->switcher->Init();
    impl_->switcher->Start();
  }
  impl_->plant = std::make_unique<PlantServer>(impl_->state);
  impl_->plant->Init();
  impl_->plant->Start();
}

PlantVendor::~PlantVendor() = default;

bool PlantVendor::owned() const noexcept { return impl_->state->owned.load(); }

int PlantVendor::fsm_id() const noexcept { return impl_->state->fsm_id.load(); }

float PlantVendor::hoist_gain() const noexcept {
  return impl_->state->hoist_gain.load(std::memory_order_relaxed);
}

void PlantVendor::set_hoist_gain(float gain) noexcept {
  impl_->state->hoist_gain.store(gain, std::memory_order_relaxed);
}

int PlantVendor::hoist_mode() const noexcept {
  return impl_->state->hoist_mode.load();
}

bool PlantVendor::hoisted() const noexcept {
  return impl_->state->hoist_mode.load() == kHoistHoisted;
}

std::uint64_t PlantVendor::hoist_generation() const noexcept {
  return impl_->state->hoist_generation.load();
}

std::uint64_t PlantVendor::reset_generation() const noexcept {
  return impl_->state->reset_generation.load();
}

void PlantVendor::mark_reset_applied(std::uint64_t generation) noexcept {
  impl_->state->reset_applied_generation.store(generation);
}

const std::string& PlantVendor::service_name() const noexcept {
  return impl_->state->service_name;
}

void PlantVendor::hoist() noexcept {
  impl_->state->hoist_generation.fetch_add(1);
  impl_->state->hoist_mode.store(kHoistHoisted);
}

void PlantVendor::lower() noexcept {
  impl_->state->hoist_mode.store(kHoistLowered);
}

void PlantVendor::slack() noexcept {
  impl_->state->hoist_mode.store(kHoistSlack);
}

void PlantVendor::reset() noexcept {
  impl_->state->fsm_id.store(1);
  impl_->state->owned.store(true);
  impl_->state->hoist_generation.fetch_add(1);
  impl_->state->hoist_mode.store(kHoistHoisted);
  impl_->state->reset_generation.fetch_add(1);
}

struct PlantClient::Impl {
  PlantClientStub client;
};

PlantClient::PlantClient(const std::string& network_interface, int dds_domain,
                         float timeout_seconds) {
  if (!(timeout_seconds > 0.0F)) {
    throw std::runtime_error("plant client timeout must be positive");
  }
  // Factory first, then the SDK client (see G1LocoClient).
  ensure_channel_factory(dds_domain, network_interface);
  impl_ = std::make_unique<Impl>();
  impl_->client.SetTimeout(timeout_seconds);
  impl_->client.Init();
}

PlantClient::~PlantClient() = default;

void PlantClient::hoist() { check(impl_->client.Hoist(), "hoist"); }

void PlantClient::lower() { check(impl_->client.Lower(), "lower"); }

void PlantClient::slack() { check(impl_->client.Slack(), "slack"); }

void PlantClient::reset() { check(impl_->client.Reset(), "reset"); }

std::string PlantClient::status() {
  std::string data;
  check(impl_->client.Status(data), "status");
  return data;
}

}  // namespace ec_native
