#include "regression/linear_regression.h"
#include "core/utils.h"

namespace mlfs {
/**
 * Creates the Linear Regression Model with validated options
 */
LinearRegression::LinearRegression(int feature_size,
                                   LinearRegressionOptions options = {})
    : opts_(std::move(options)) {
  opts_.validate();

  weights_ = Eigen::MatrixXd::Random(feature_size, opts_.batch_size);
  bias_ = uniform<double>(-1.0, 1.0);
}

// TODO: Implement
void LinearRegression::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
}

// TODO: Implement
Eigen::VectorXd LinearRegression::predict(const Eigen::MatrixXd &X) const {
  return Eigen::VectorXd();
}

} // namespace mlfs