#include "regression/linear_regression.h"
#include "core/utils.h"
#include <algorithm>
#include <iostream>

namespace mlfs {
/**
 * Creates the Linear Regression Model with validated options
 */
LinearRegression::LinearRegression(int feature_size,
                                   LinearRegressionOptions options = {})
    : opts_(std::move(options)) {
  opts_.validate();

  weights_ = Eigen::VectorXd::Random(feature_size);
  bias_ = mlfs::uniform<double>(-1., 1.);
}

void LinearRegression::fit(const mlfs::RowMatrixXd &X,
                           const Eigen::VectorXd &Y) {
  fit_gd(X, Y);
}

Eigen::VectorXd LinearRegression::predict(const mlfs::RowMatrixXd &X) const {
  return ((X * weights_).array() + bias_).matrix();
}

void LinearRegression::fit_closed_form(const mlfs::RowMatrixXd &X,
                                       const Eigen::VectorXd &y) {}

void LinearRegression::fit_gd(const mlfs::RowMatrixXd &X,
                              const Eigen::VectorXd &Y) {
  const int total_rows = X.rows();
  std::cout << total_rows << "\n";

  for (int epoch = 0; epoch < opts_.epochs; epoch++) {
    // Testing run for one epoch
    for (int i = 0; i < total_rows; i += opts_.batch_size) {
      int current_batch_size = std::min(opts_.batch_size, total_rows - i);
      RowMatrixXd batch_X = X.middleRows(i, current_batch_size);
      Eigen::VectorXd batch_Y = Y.middleRows(i, current_batch_size);

      // Make preduction and calculate loss
      Eigen::VectorXd y = predict(batch_X);
      double L = opts_.loss->compute(batch_Y, y);

      // Output Loss
      std::cout << "[" << i << " - " << i + current_batch_size
                << "] - Loss: " << L << "\n";

      double scale = -2.0 / opts_.batch_size;
      // Calculate gradients
      Eigen::VectorXd dW = (batch_X.transpose() * (batch_Y - y)) * scale;
      double dB = (batch_Y - y).sum() * scale;

      // Update Weights and biases
      weights_ = weights_ - opts_.learning_rate * dW;
      bias_ = bias_ - opts_.learning_rate * dB;
    }
  }
  std::cout << "W: " << weights_ << ", B: " << bias_ << "\n";
}

} // namespace mlfs