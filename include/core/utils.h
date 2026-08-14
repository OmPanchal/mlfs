#pragma once
#include <random>

namespace mlfs {
template <typename T> inline T uniform(double min, double max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dis(min, max);
  return dis(gen);
}
} // namespace mlfs