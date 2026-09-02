#include "regression/polynomial_regression.h"
#include "core/types.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// TODO: The inner workins of Polynomial Regression is identical to Linear
//  Regression, should unify these inner workings to a single method

namespace mlfs {
RowMatrixXd create_vandermonde_matrix(RowMatrixXd x, int degree) {
  // Make sure that the input is a matrix with column size 1
  if (x.cols() != 1) {
    throw std::invalid_argument("");
  }
  // Create the output matrix
  int n = x.rows();
  RowMatrixXd V(n, degree + 1);

  // Set the first column to one
  V.col(0).setOnes();
  V.col(1) = x;

  for (size_t i = 1; i < degree + 1; i++) {
    V.col(i) = V.col(i - 1).cwiseProduct(V.col(1));
  }
  return V;
}

/**
 * Creates the Polynomial Regression Model with validated options
 */
PolynomialRegression::PolynomialRegression(
    int degree, PolynomialRegressionOptions options = {})
    : opts_(std::move(options)), degree_(degree) {

  weights_ = Eigen::VectorXd::Random(degree + 1);
}

void PolynomialRegression::fit(mlfs::CSVDataset &data) {
  RowMatrixXd V_x = create_vandermonde_matrix(data.get_features(), degree_);

  if (opts_.solver == CF) {
    fit_closed_form(V_x, data.get_target());
  } else if (opts_.solver == GD) {
    fit_gd(V_x, data.get_target());
  } else {
    throw std::invalid_argument(SOLVER_DOES_NOT_EXIST);
  }
}

Eigen::VectorXd
PolynomialRegression::predict(const mlfs::RowMatrixXd &X) const {
  RowMatrixXd V_x = create_vandermonde_matrix(X, degree_);
  return V_x * weights_;
}

Eigen::VectorXd
PolynomialRegression::predict_(const mlfs::RowMatrixXd &X) const {
  // Fast predict which will only need to calculate the vandermonde matrix once
  return X * weights_;
}

void PolynomialRegression::fit_closed_form(const mlfs::RowMatrixXd &X,
                                           const Eigen::VectorXd &Y) {
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
    weights_ = (Z + R).inverse() * X.transpose() * Y;
  }
}

void PolynomialRegression::fit_gd(const mlfs::RowMatrixXd &X,
                                  const Eigen::VectorXd &Y) {
  const int total_rows = X.rows();

  for (size_t epoch = 1; epoch < opts_.epochs + 1; epoch++) {
    for (int i = 0; i < total_rows; i += opts_.batch_size) {
      int current_batch_size = std::min(opts_.batch_size, total_rows - i);
      RowMatrixXd batch_X = X.middleRows(i, current_batch_size);
      Eigen::VectorXd batch_Y = Y.middleRows(i, current_batch_size);

      // X will already by a vandermonde matrix so use predict_
      Eigen::VectorXd y = predict_(batch_X);

      // Calculate loss and regularisation gradients
      Eigen::VectorXd l1_grad =
          (weights_.array().sign() * opts_.alpha).matrix();

      Eigen::VectorXd l2_grad =
          (weights_.array() * 2 * (1 - opts_.alpha)).matrix();

      Eigen::VectorXd regularisation_grad = l1_grad + l2_grad;

      // remove affect of gradient to the bias
      regularisation_grad(regularisation_grad.size() - 1) = 0.;

      // calculate the gradient
      Eigen::VectorXd dW =
          batch_X.transpose() *
              opts_.loss->gradient(batch_Y, y, batch_Y.rows()) +
          (opts_.lambda * regularisation_grad);

      weights_ = weights_ - opts_.learning_rate * dW;
    }
  }
}

}; // namespace mlfs
