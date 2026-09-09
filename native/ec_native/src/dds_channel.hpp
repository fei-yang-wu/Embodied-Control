#pragma once

#include <mutex>
#include <stdexcept>
#include <string>

#include <unitree/robot/channel/channel_factory.hpp>

namespace ec_native {

// ChannelFactory::Instance()->Init is a process-wide singleton guarded by its
// own mInited flag: a second call with different arguments is silently
// ignored, not rejected. A G1 session builds two DDS participants in one
// process (the low-level backend and the sport-service client), so the failure
// mode is real - initialize on `lo` for a plant rehearsal, then reach for the
// robot on eth0, and the second component quietly talks to the wrong
// transport. Record the first (domain, interface) and reject a conflicting one.
inline void ensure_channel_factory(int domain, const std::string& interface) {
  static std::mutex mutex;
  static bool initialized = false;
  static int active_domain = 0;
  static std::string active_interface;

  const std::lock_guard<std::mutex> guard(mutex);
  if (initialized) {
    if (domain != active_domain || interface != active_interface) {
      throw std::runtime_error(
          "DDS channel factory is already initialized on domain " +
          std::to_string(active_domain) + " interface '" + active_interface +
          "'; cannot reinitialize on domain " + std::to_string(domain) +
          " interface '" + interface + "'");
    }
    return;
  }
  unitree::robot::ChannelFactory::Instance()->Init(domain, interface);
  initialized = true;
  active_domain = domain;
  active_interface = interface;
}

}  // namespace ec_native
