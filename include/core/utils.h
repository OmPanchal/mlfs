#pragma once
#include <random>

namespace mlfs {
/**
 * Returns a random value between min and max from a uniform distribution
 * @param min The minimum value of the sample
 * @param max The maximum value of the sample
 * @return A uniformly random number between min and max
 */
template <typename T> inline T uniform(double min, double max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dis(min, max);
  return dis(gen);
}
} // namespace mlfs