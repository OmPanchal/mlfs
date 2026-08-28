#include "regression/linear_regression.h"
#include "core/errors.h"
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

void LinearRegression::fit(mlfs::CSVDataset &dataset) {
  if (opts_.solver == SolverType::GD) {
    fit_gd(dataset.get_raw_features(), dataset.get_target());
  } else if (opts_.solver == SolverType::CF) {
    fit_closed_form(dataset.get_raw_features(), dataset.get_target());
  }
}

Eigen::VectorXd LinearRegression::predict(const mlfs::RowMatrixXd &X) const {
  return ((X * weights_).array()).matrix();
}

void LinearRegression::fit_closed_form(const mlfs::RowMatrixXd &X,
                                       const Eigen::VectorXd &y) {

  // Closed form solution does not exist for l1 regularisation
  if (opts_.alpha != 0) {
    throw std::runtime_error(REGRESSION_NO_SOLUTION);
  }

  // Matrix to be inversed
  RowMatrixXd Z = X.transpose() * X;
  RowMatrixXd R = (opts_.lambda * X.rows() *
                   Eigen::MatrixXd::Identity(weights_.rows(), weights_.rows()));
  // Make the final diagonal of the regularisation gradient to zero to make it
  // not affect the bias
  R(R.rows() - 1, R.cols() - 1) = 0.;

  // Check if the inverse is possible or not
  if ((Z + R).determinant() == 0) {
    throw std::runtime_error(MATRIX_NOT_INVERTIBLE);
  } else {
    weights_ = (Z + R).inverse() * X.transpose() * y;
  }
}

void LinearRegression::fit_gd(const mlfs::RowMatrixXd &X,
                              const Eigen::VectorXd &Y) {
  const int total_rows = X.rows();

  for (int epoch = 1; epoch < opts_.epochs + 1; epoch++) {
    // Split into batches
    for (int i = 0; i < total_rows; i += opts_.batch_size) {
      // Split the dataset into batches
      int current_batch_size = std::min(opts_.batch_size, total_rows - i);
      RowMatrixXd batch_X = X.middleRows(i, current_batch_size);
      Eigen::VectorXd batch_Y = Y.middleRows(i, current_batch_size);

      // Make prediction on the batch
      Eigen::VectorXd y = predict(batch_X);

      // Calculate loss and regularisation gradients
      Eigen::VectorXd l1_grad =
          (weights_.array().sign() * opts_.alpha).matrix();

      Eigen::VectorXd l2_grad =
          (weights_.array() * 2 * (1 - opts_.alpha)).matrix();

      Eigen::VectorXd regularisation_grad = l1_grad + l2_grad;

      // remove affect of gradient to the bias
      regularisation_grad(regularisation_grad.size() - 1) = 0.;

      Eigen::VectorXd dW =
          batch_X.transpose() *
              opts_.loss->gradient(batch_Y, y, batch_Y.rows()) +
          (opts_.lambda * regularisation_grad);

      // Update Weights and biases
      weights_ = weights_ - opts_.learning_rate * dW;
    }
  }
}

} // namespace mlfs