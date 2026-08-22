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
}

void LinearRegression::fit(const mlfs::RowMatrixXd &X,
                           const Eigen::VectorXd &Y) {
  if (opts_.solver == SolverType::GD) {
    fit_gd(X, Y);
  } else if (opts_.solver == SolverType::CF) {
    fit_closed_form(X, Y);
  }
}

Eigen::VectorXd LinearRegression::predict(const mlfs::RowMatrixXd &X) const {
  return ((X * weights_).array()).matrix();
}

void LinearRegression::fit_closed_form(const mlfs::RowMatrixXd &X,
                                       const Eigen::VectorXd &y) {
  // Check if the inverse is possible or not
  RowMatrixXd Z = (X.transpose() * X);
  if (Z.determinant() == 0) {
    throw std::runtime_error(
        "Closed form solution not possible due to non invertable matrix");
  } else {
    weights_ = Z.inverse() * X.transpose() * y;
  }
}

void LinearRegression::fit_gd(const mlfs::RowMatrixXd &X,
                              const Eigen::VectorXd &Y) {
  const int total_rows = X.rows();

  for (int epoch = 0; epoch < opts_.epochs; epoch++) {
    // Split into batches
    std::cout << "(EPOCH: " << epoch << ") - ";
    for (int i = 0; i < total_rows; i += opts_.batch_size) {
      int current_batch_size = std::min(opts_.batch_size, total_rows - i);
      RowMatrixXd batch_X = X.middleRows(i, current_batch_size);
      Eigen::VectorXd batch_Y = Y.middleRows(i, current_batch_size);

      // Make preduction and calculate loss with regularisation losses
      Eigen::VectorXd y = predict(batch_X);
      double L = opts_.loss->compute(batch_Y, y) +
                 opts_.lambda * (opts_.alpha * weights_.lpNorm<1>() +
                                 (1 - opts_.alpha) * weights_.squaredNorm());

      // Output Loss
      std::cout << "[" << i << " - " << i + current_batch_size
                << "] - Loss: " << L << " -- ";

      // Calculate gradients
      Eigen::VectorXd l1_grad =
          ((weights_.array() / (weights_.array().square() + 0.000001).sqrt()) *
           opts_.alpha)
              .matrix();

      Eigen::VectorXd l2_grad =
          (weights_.array() * 2 * (1 - opts_.alpha)).matrix();

      Eigen::VectorXd dW =
          batch_X.transpose() *
              opts_.loss->gradient(batch_Y, y, opts_.batch_size) +
          opts_.lambda * (l1_grad + l2_grad);

      // Update Weights and biases
      weights_ = weights_ - opts_.learning_rate * dW;
    }
    std::cout << "\n";
  }
}

} // namespace mlfs