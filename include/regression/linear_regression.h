#pragma once

#include "core/model.h"
#include "core/types.h"
#include <Eigen/Dense>

namespace mlfs {
class LinearRegression : public Model {
public:
  /**
   * Initialises the model with the initial options
   * @param LinearRegressionOptions The option struct
   */
  LinearRegression(int feature_size, LinearRegressionOptions opts_);
  ~LinearRegression() = default;

  void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
  Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;

  // Getters
  [[nodiscard]] const Eigen::MatrixXd &getWeights() const { return weights_; }
  [[nodiscard]] double getBias() const { return bias_; }
  [[nodiscard]] const LinearRegressionOptions &getOpts() const { return opts_; }

private:
  Eigen::MatrixXd weights_;
  double bias_;
  const LinearRegressionOptions opts_;

  /**
   * Trains the model on features and targerts using a closed form approach
   * @param X The features
   * @param Y The target values
   */
  void fit_closed_form(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);

  /**
   * Trains the model on features and targerts using a stochastic gradient
   * descent approach
   * @param X The features
   * @param Y The target values
   */
  void fit_sgd(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
};
} // namespace mlfs