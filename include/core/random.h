#pragma once
#include <random>

namespace mlfs {

class Random {
public:
  static std::mt19937 &get_engine() {
    thread_local std::mt19937 engine([]() {
      std::random_device rd;
      return std::mt19937(rd());
    }());

    return engine;
  }

  static double uniform(double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(get_engine());
  }
};
} // namespace mlfs