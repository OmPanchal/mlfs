#include "regression/logistic_regression.h"
#include "core/random.h"

namespace mlfs {
LogisticRegression::LogisticRegression(int feature_size,
                                       RegressionOptions options = {})
    : opts_(std::move(options)) {
  opts_.validate();

  weights_ = Eigen::VectorXd::NullaryExpr(
      feature_size, []() { return Random::uniform(-1.0, 1.0); });
  bias_ = Random::uniform(-1.0, 1.0);
}

void LogisticRegression::fit(mlfs::CSVDataset &dataset) {
  if (opts_.solver == SolverType::GD) {
    fit_gd(dataset.get_features(), dataset.get_target());
  } else {
    throw std::invalid_argument(SOLVER_DOES_NOT_EXIST);
  }
}

Eigen::VectorXd LogisticRegression::predict(const mlfs::RowMatrixXd &X) const {
  return sigmoid(((X * weights_).array() + bias_).matrix());
}

void LogisticRegression::fit_gd(const mlfs::RowMatrixXd &X,
                                const Eigen::VectorXd &Y) {}
} // namespace mlfs