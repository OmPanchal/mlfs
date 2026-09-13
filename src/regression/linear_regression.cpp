#include "regression/linear_regression.h"
#include "core/errors.h"
#include "core/random.h"
#include "core/utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace mlfs {
/**
 * Creates the Linear Regression Model with validated options
 */
LinearRegression::LinearRegression(int feature_size,
                                   LinearRegressionOptions options = {})
    : opts_(std::move(options)) {
  opts_.validate();

  weights_ = Eigen::VectorXd::NullaryExpr(
      feature_size, []() { return Random::uniform(-1.0, 1.0); });
  bias_ = Random::uniform(-1.0, 1.0);

  std::cout << "\n\n PARAMS: " << weights_ << " " << bias_ << "\n\n";
}

void LinearRegression::fit(mlfs::CSVDataset &dataset) {
  if (opts_.solver == SolverType::GD) {
    fit_gd(dataset.get_raw_features(), dataset.get_target());
  } else if (opts_.solver == SolverType::CF) {
    fit_closed_form(dataset.get_raw_features(), dataset.get_target());
  } else {
    throw std::invalid_argument(SOLVER_DOES_NOT_EXIST);
  }
}

Eigen::VectorXd LinearRegression::predict(const mlfs::RowMatrixXd &X) const {
  return ((X * weights_).array() + bias_).matrix();
}

void LinearRegression::fit_closed_form(const mlfs::RowMatrixXd &X,
                                       const Eigen::VectorXd &Y) {
  // ? Consider Singular Value Decomposition
  // Closed form solution does not exist for l1 regularisation
  if (opts_.alpha != 0) {
    throw std::runtime_error(REGRESSION_NO_SOLUTION);
  }

  // Make sure that the shape of the input and the weights matches
  if (X.cols() != weights_.rows()) {
    throw std::runtime_error(FEATURE_WEIGHT_SIZE_MISMATCH);
  }

  // Matrix to be inversed
  RowMatrixXd Z = X.transpose() * X;
  RowMatrixXd R = (opts_.lambda * X.rows() *
                   Eigen::MatrixXd::Identity(weights_.rows(), weights_.rows()));

  // Check if the inverse is possible or not
  if ((Z + R).determinant() == 0) {
    throw std::runtime_error(MATRIX_NOT_INVERTIBLE);
  } else {
    weights_ = (Z + R).inverse() * X.transpose() * Y;
  }
}

void LinearRegression::fit_gd(const mlfs::RowMatrixXd &X,
                              const Eigen::VectorXd &Y) {
  const int total_rows = X.rows();

  // Make sure that the shape of the input and the weights matches
  if (X.cols() != weights_.rows()) {
    throw std::runtime_error(FEATURE_WEIGHT_SIZE_MISMATCH);
  }

  for (int epoch = 1; epoch < opts_.epochs + 1; epoch++) {
    // Split into batches
    for (int i = 0; i < total_rows; i += opts_.batch_size) {
      // Split the dataset into batches
      int current_batch_size = std::min(opts_.batch_size, total_rows - i);
      RowMatrixXd batch_X = X.middleRows(i, current_batch_size);
      Eigen::VectorXd batch_Y = Y.middleRows(i, current_batch_size);

      // Make prediction on the batch
      Eigen::VectorXd y = predict(batch_X);

      // Calculate regularisation gradients
      Eigen::VectorXd l1_grad =
          opts_.alpha * l1_regulariser->gradient(weights_);

      Eigen::VectorXd l2_grad =
          (1 - opts_.alpha) * l2_regulariser->gradient(weights_);

      Eigen::VectorXd regularisation_grad = l1_grad + l2_grad;

      // Calculate weights and biases' gradient
      Eigen::VectorXd dW =
          batch_X.transpose() *
              opts_.loss->gradient(batch_Y, y, batch_Y.rows()) +
          (opts_.lambda * regularisation_grad);

      double dB = opts_.loss->gradient(batch_Y, y, batch_Y.rows()).sum();

      // Update Weights and biases
      weights_ = weights_ - opts_.learning_rate * dW;
      bias_ = bias_ - opts_.learning_rate * dB;
    }
  }
}

} // namespace mlfs