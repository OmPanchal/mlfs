#pragma once
#include <Eigen/Dense>

namespace mlfs {

/**
 * The base Norm Regulariser class
 */
class NormRegulariser {
public:
  virtual ~NormRegulariser() = default;

  /**
   * Computes the regularisation penalty
   * @param weights The model weights
   * @return The corresponding weight penalty
   */
  inline virtual double compute(Eigen::VectorXd &weights) const = 0;

  /**
   * Computes the gradient of the regularisation penalty
   * @param weights The model weights
   * @return The derivative of the regularisation penalty with respect to the
   * weights
   */
  inline virtual Eigen::VectorXd gradient(Eigen::VectorXd &weights) const = 0;
};

class L1Regulariser : public NormRegulariser {
public:
  L1Regulariser() = default;
  ~L1Regulariser() = default;

  inline double compute(Eigen::VectorXd &weights) const override {
    return weights.lpNorm<1>();
  }

  inline Eigen::VectorXd gradient(Eigen::VectorXd &weights) const override {
    return weights.cwiseSign();
  }
};

class L2Regulariser : public NormRegulariser {
public:
  L2Regulariser() = default;
  ~L2Regulariser() = default;

  inline double compute(Eigen::VectorXd &weights) const override {
    return weights.norm();
  }

  inline Eigen::VectorXd gradient(Eigen::VectorXd &weights) const override {
    return weights * 2;
  }
};

} // namespace mlfs