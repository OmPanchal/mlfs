#include "regression/polynomial_regression.h"
#include "core/types.h"

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

PolynomialRegression::PolynomialRegression(
    int feature_size, int degree, PolynomialRegressionOptions options = {})
    : opts_(std::move(options)), degree_(degree) {

  weights_ = Eigen::VectorXd::Random(feature_size + 1);
}

Eigen::VectorXd
PolynomialRegression::predict(const mlfs::RowMatrixXd &X) const {
  RowMatrixXd V_x = create_vandermonde_matrix(X, degree_);
  return V_x * weights_;
}

Eigen::VectorXd
PolynomialRegression::predict_(const mlfs::RowMatrixXd &X) const {
  return X * weights_;
}
}; // namespace mlfs
