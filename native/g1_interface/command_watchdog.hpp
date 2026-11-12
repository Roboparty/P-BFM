#pragma once

#include <chrono>

// Access under the command mutex. A late producer cannot clear a timeout.
class CommandWatchdog {
 public:
  using Clock = std::chrono::steady_clock;
  static constexpr auto timeout = std::chrono::milliseconds(100);

  bool Accept(Clock::time_point now) {
    if (armed_ && !CanPublish(now)) return false;
    last_update_ = now;
    armed_ = true;
    return true;
  }

  bool CanPublish(Clock::time_point now) {
    if (armed_ && (now < last_update_ || now - last_update_ >= timeout)) {
      fault_ = true;
    }
    return armed_ && !fault_;
  }

  bool fault() const { return fault_; }

 private:
  Clock::time_point last_update_{};
  bool armed_ = false;
  bool fault_ = false;
};
