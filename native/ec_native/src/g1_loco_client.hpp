#pragma once

#include <memory>
#include <string>

namespace ec_native {

struct G1LocoStatus {
  int fsm_id = -1;
  int fsm_mode = -1;
  int balance_mode = -1;
  float stand_height = 0.0F;
  float swing_height = 0.0F;
};

// Thin RAII wrapper over the SDK's g1::LocoClient (the "sport" service) - the
// high-level command axis, i.e. what the operator's joystick drives. Every
// call is a blocking DDS RPC that can take its whole timeout to return, so
// this must never be touched from the 500 Hz writer thread or the control
// thread; it is operator-rate only.
//
// The SDK returns int32_t status codes. This wrapper throws on non-zero
// instead, so a caller that forgets to check a return value cannot silently
// continue after a refused takeover.
class G1LocoClient {
 public:
  G1LocoClient(const std::string& network_interface, int dds_domain,
               float timeout_seconds);
  ~G1LocoClient();

  G1LocoClient(const G1LocoClient&) = delete;
  G1LocoClient& operator=(const G1LocoClient&) = delete;

  int fsm_id() const;
  int fsm_mode() const;
  int balance_mode() const;
  float stand_height() const;
  float swing_height() const;
  G1LocoStatus status() const;

  void set_fsm_id(int fsm_id);
  void zero_torque();
  void damp();
  void squat();
  void sit();
  void stand_up();
  void start();

  void balance_stand();
  void continuous_gait(bool enabled);
  void set_stand_height(float height);
  void high_stand();
  void low_stand();
  void set_swing_height(float height);
  void set_speed_mode(int speed_mode);

  void move(float vx, float vy, float vyaw, bool continuous);
  void stop_move();

  void wave_hand(bool turn);
  void shake_hand(int stage);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ec_native
