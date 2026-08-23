#include "regression/linear_regression.h"
#include "core/utils.h"
#include <algorithm>
#include <iostream>

/** TODO: Make the regularisation weight updates only affect the weights rather
 */

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

  // Closed form solution does not exist for l1 regularisation
  if (opts_.alpha != 0) {
    throw std::runtime_error(
        "[Error] A closed form solution does not exist "
        "for the model with alpha > 0 (i.e. a model with l1 regularisation). "
        "Please try using gradient descent instead.");
  }

  // Matrix to be inversed
  RowMatrixXd Z =
      ((X.transpose() * X) +
       (opts_.lambda * X.rows() *
        Eigen::MatrixXd::Identity(weights_.rows(), weights_.rows())));

  // Check if the inverse is possible or not
  if (Z.determinant() == 0) {
    throw std::runtime_error(
        "[Error] Closed form solution not possible due to "
        "non invertable matrix. Please try using gradient descent instead.");
  } else {
    weights_ = Z.inverse() * X.transpose() * y;
  }
}

void LinearRegression::fit_gd(const mlfs::RowMatrixXd &X,
                              const Eigen::VectorXd &Y) {
  const int total_rows = X.rows();

  for (int epoch = 1; epoch < opts_.epochs + 1; epoch++) {
    // Split into batches
    std::cout << "(EPOCH: " << epoch << ") - ";
    for (int i = 0; i < total_rows; i += opts_.batch_size) {
      // Split the dataset into batches
      int current_batch_size = std::min(opts_.batch_size, total_rows - i);
      RowMatrixXd batch_X = X.middleRows(i, current_batch_size);
      Eigen::VectorXd batch_Y = Y.middleRows(i, current_batch_size);

      // Make prediction and calculate loss with regularisation losses
      Eigen::VectorXd y = predict(batch_X);
      double L = opts_.loss->compute(batch_Y, y) +
                 opts_.lambda * (opts_.alpha * weights_.lpNorm<1>() +
                                 (1 - opts_.alpha) * weights_.squaredNorm());

      // Output Loss
      std::cout << "[" << i << " - " << i + current_batch_size
                << "] - Loss: " << L << " -- ";

      // TODO: Optimise implementation which applies the gradient change to just
      // the weights Calculate gradients
      Eigen::VectorXd l1_grad =
          (weights_.array().sign() * opts_.alpha).matrix();

      Eigen::VectorXd l2_grad =
          (weights_.array() * 2 * (1 - opts_.alpha)).matrix();

      Eigen::VectorXd dW =
          batch_X.transpose() *
              opts_.loss->gradient(batch_Y, y, batch_Y.rows()) +
          opts_.lambda * (l1_grad + l2_grad);

      // Update Weights and biases
      weights_ = weights_ - opts_.learning_rate * dW;
    }
    std::cout << "\n";
  }
}

} // namespace mlfs