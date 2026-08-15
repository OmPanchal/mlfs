#pragma once
#include "core/losses.h"
#include <memory>

namespace mlfs {
/**
 * Solver type for different .fit approaches
 */
enum SolverType {
  CF, // Closed Form
  SGD // Stochastic Gradient Descent
};

struct LinearRegressionOptions {
  double learning_rate = 0.01;
  int epochs = 1000;
  int batch_size = 1;
  std::unique_ptr<Loss> loss = std::make_unique<MSE>();
  SolverType solver = SolverType::CF;

  void validate() const {
    if (learning_rate <= 0.0) {
      throw std::invalid_argument(
          "[Invalid Argument] learning_rate must be positive");
    }
    if (epochs <= 0) {
      throw std::invalid_argument("[Invalid Argument] epochs must be positive");
    }
    if (batch_size != 1) {
      throw std::runtime_error("[Not Implemented] Only SGD is supported");
    }
  }
};

using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

} // namespace mlfs