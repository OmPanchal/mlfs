#pragma once
#include <Eigen/Dense>

namespace mlfs {
/**
 * Base Loss Interface
 */
class Loss {
public:
  virtual ~Loss() = default;
  /**
   * Computes the loss given the model prediction and the target
   * @param Y The target value
   * @param Y_pred The model's prediction
   * @return The result of the loss function
   */
  virtual double compute(Eigen::VectorXd &Y, Eigen::VectorXd &Y_pred) const = 0;
};

class MSE : public Loss {
public:
  MSE() = default;
  ~MSE() = default;

  double compute(Eigen::VectorXd &Y, Eigen::VectorXd &Y_pred) const override;
};
} // namespace mlfs