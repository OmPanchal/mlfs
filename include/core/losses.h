#pragma once
#include <Eigen/Dense>

namespace mlfs {
using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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
  inline virtual double compute(Eigen::VectorXd &Y,
                                Eigen::VectorXd &Y_pred) const = 0;

  /**
   * Returns the gradient of the loss function
   * @param Y The target value
   * @param Y_pred The model's prediction
   * @return The derivative of the loss funciton with respect to the weights
   */
  inline virtual RowMatrixXd gradient(Eigen::VectorXd &Y,
                                      Eigen::VectorXd &Y_pred,
                                      int batch_size) const = 0;
};

class MSE : public Loss {
public:
  MSE() = default;
  ~MSE() = default;

  inline double compute(Eigen::VectorXd &Y,
                        Eigen::VectorXd &Y_pred) const override {
    return (Y - Y_pred).squaredNorm() / Y.size();
  };

  inline virtual RowMatrixXd gradient(Eigen::VectorXd &Y,
                                      Eigen::VectorXd &Y_pred,
                                      int batch_size) const override {
    return (-2.0 / batch_size) * (Y - Y_pred);
  };
};
} // namespace mlfs