#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace ec_native {

// The vendor side of the simulated G1: the "sport" and "motion_switcher" RPC
// services the real robot runs, served with the SDK's own Server class so the
// unchanged SDK clients (G1Runtime, the backend's MotionSwitcherClient) talk
// to the plant exactly as they talk to hardware. Plus a plant-only "ec_plant"
// service for what hardware has no RPC for: the hoist.
//
// Ownership is the whole point. While `owned()` the vendor drives the joints
// and the plant rejects rt/lowcmd; ReleaseMode hands them over, SelectMode
// takes them back. ReleaseMode is refused unless the vendor FSM is damp (1),
// the same rule the hardware lifecycle is built around.
class PlantVendor {
 public:
  PlantVendor(const std::string& service_name, bool vendor_enabled,
              bool hoisted);
  ~PlantVendor();

  PlantVendor(const PlantVendor&) = delete;
  PlantVendor& operator=(const PlantVendor&) = delete;

  // Gantry states: 1 hoisted (rigid hold at the captured pose), 2 lowered
  // (feet carry the weight, the strap only catches a drop and steadies the
  // tilt), 0 slack (strap paid out, nothing holds the robot).
  static constexpr int kHoistSlack = 0;
  static constexpr int kHoistHoisted = 1;
  static constexpr int kHoistLowered = 2;

  bool owned() const noexcept;
  int fsm_id() const noexcept;
  int hoist_mode() const noexcept;
  bool hoisted() const noexcept;
  // Bumps on every hoist request, so the plant re-captures the hoist target
  // at the robot's current pose rather than the one it started at.
  std::uint64_t hoist_generation() const noexcept;
  const std::string& service_name() const noexcept;

  // In-process controls, for tests and for a plant driven from Python.
  void hoist() noexcept;
  void lower() noexcept;
  void slack() noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Client for the plant-only "ec_plant" service (hoist / lower / status).
class PlantClient {
 public:
  PlantClient(const std::string& network_interface, int dds_domain,
              float timeout_seconds);
  ~PlantClient();

  PlantClient(const PlantClient&) = delete;
  PlantClient& operator=(const PlantClient&) = delete;

  void hoist();
  void lower();
  void slack();
  // JSON: {"owned":bool,"fsm_id":int,"hoisted":bool,"hoist_mode":int,"service":str}
  std::string status();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ec_native
