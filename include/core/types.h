#pragma once
#include "core/losses.h"
#include <memory>

namespace mlfs {
/**
 * Solver type for different .fit approaches
 */
enum SolverType {
  CF, // Closed Form
  GD  // Gradient Descent
};

/**
 * Linear Regression Options struct
 */
struct LinearRegressionOptions {
  double learning_rate = 0.01;
  int epochs = 1000;
  int batch_size = 1;
  std::unique_ptr<Loss> loss = std::make_unique<MSE>();
  SolverType solver = SolverType::CF;
  double lambda = 0;
  double alpha = 0;

  void validate() const {
    if (learning_rate <= 0.0) {
      throw std::invalid_argument(
          "[Invalid Argument] learning_rate must be positive");
    }
    if (epochs <= 0) {
      throw std::invalid_argument("[Invalid Argument] epochs must be positive");
    }
    if (batch_size < 1) {
      throw std::invalid_argument(
          "[Invalid Argument] batch_size must be positive");
    }
    if (lambda < 0) {
      throw std::invalid_argument("[Invalid Argument] lambda must be positive");
    }
    if (alpha < 0 || alpha > 1) {
      throw std::invalid_argument(
          "[Invalid Argument] alpha must be in the interval [0, 1]");
    }
  }
};

/**
 * CSV Loader Options
 * */
struct CSVLoaderOptions {
  bool has_header = true;
  char separator = ',';
};

/**
 * The Row Major Matrix type
 */
using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

} // namespace mlfs