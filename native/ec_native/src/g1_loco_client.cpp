#include "g1_loco_client.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

#include <unitree/robot/g1/loco/g1_loco_client.hpp>

#include "dds_channel.hpp"

namespace ec_native {
namespace {

void check(std::int32_t status, const char* verb) {
  if (status != 0) {
    throw std::runtime_error("G1 sport service refused " + std::string(verb) +
                             " (status " + std::to_string(status) + ")");
  }
}

}  // namespace

struct G1LocoClient::Impl {
  unitree::robot::g1::LocoClient client;
};

G1LocoClient::G1LocoClient(const std::string& network_interface,
                           int dds_domain, float timeout_seconds) {
  if (!(timeout_seconds > 0.0F)) {
    throw std::runtime_error("G1 sport service timeout must be positive");
  }
  // The SDK client opens its DDS channels in its constructor, so the factory
  // must be initialized before Impl exists: constructing it in the
  // initializer list segfaults in a process where this is the first DDS
  // object.
  ensure_channel_factory(dds_domain, network_interface);
  impl_ = std::make_unique<Impl>();
  impl_->client.SetTimeout(timeout_seconds);
  impl_->client.Init();
}

G1LocoClient::~G1LocoClient() = default;

int G1LocoClient::fsm_id() const {
  int value = -1;
  check(impl_->client.GetFsmId(value), "GetFsmId");
  return value;
}

int G1LocoClient::fsm_mode() const {
  int value = -1;
  check(impl_->client.GetFsmMode(value), "GetFsmMode");
  return value;
}

int G1LocoClient::balance_mode() const {
  int value = -1;
  check(impl_->client.GetBalanceMode(value), "GetBalanceMode");
  return value;
}

float G1LocoClient::stand_height() const {
  float value = 0.0F;
  check(impl_->client.GetStandHeight(value), "GetStandHeight");
  return value;
}

float G1LocoClient::swing_height() const {
  float value = 0.0F;
  check(impl_->client.GetSwingHeight(value), "GetSwingHeight");
  return value;
}

G1LocoStatus G1LocoClient::status() const {
  G1LocoStatus out;
  out.fsm_id = fsm_id();
  out.fsm_mode = fsm_mode();
  out.balance_mode = balance_mode();
  out.stand_height = stand_height();
  out.swing_height = swing_height();
  return out;
}

void G1LocoClient::set_fsm_id(int id) {
  check(impl_->client.SetFsmId(id), "SetFsmId");
}

void G1LocoClient::zero_torque() {
  check(impl_->client.ZeroTorque(), "ZeroTorque");
}

void G1LocoClient::damp() { check(impl_->client.Damp(), "Damp"); }

void G1LocoClient::squat() { check(impl_->client.Squat(), "Squat"); }

void G1LocoClient::sit() { check(impl_->client.Sit(), "Sit"); }

void G1LocoClient::stand_up() { check(impl_->client.StandUp(), "StandUp"); }

void G1LocoClient::start() { check(impl_->client.Start(), "Start"); }

void G1LocoClient::balance_stand() {
  check(impl_->client.BalanceStand(), "BalanceStand");
}

void G1LocoClient::continuous_gait(bool enabled) {
  check(impl_->client.ContinuousGait(enabled), "ContinuousGait");
}

void G1LocoClient::set_stand_height(float height) {
  check(impl_->client.SetStandHeight(height), "SetStandHeight");
}

void G1LocoClient::high_stand() {
  check(impl_->client.HighStand(), "HighStand");
}

void G1LocoClient::low_stand() { check(impl_->client.LowStand(), "LowStand"); }

void G1LocoClient::set_swing_height(float height) {
  check(impl_->client.SetSwingHeight(height), "SetSwingHeight");
}

void G1LocoClient::set_speed_mode(int speed_mode) {
  check(impl_->client.SetSpeedMode(speed_mode), "SetSpeedMode");
}

void G1LocoClient::move(float vx, float vy, float vyaw, bool continuous) {
  check(impl_->client.Move(vx, vy, vyaw, continuous), "Move");
}

void G1LocoClient::stop_move() {
  check(impl_->client.StopMove(), "StopMove");
}

void G1LocoClient::wave_hand(bool turn) {
  check(impl_->client.WaveHand(turn), "WaveHand");
}

void G1LocoClient::shake_hand(int stage) {
  check(impl_->client.ShakeHand(stage), "ShakeHand");
}

}  // namespace ec_native
