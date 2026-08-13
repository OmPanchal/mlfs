#pragma once
#include "model.h"

namespace mlfs {
enum class OptimiserType {
  CF,  // Closed Form
  BGD, // Batch Gradient Descent
};

/**
 * The base Regression Class
 */
class Regression : public Model {
public:
  Regression(OptimiserType otype);
  ~Regression() override = default;
  virtual void fit(Eigen::MatrixXd &X, Eigen::VectorXd &y) = 0;
  virtual Eigen::VectorXd predict(Eigen::MatrixXd &X) const = 0;

private:
  Eigen::MatrixXd weights_;
  double bias_;
  OptimiserType otype_;
};
} // namespace mlfs