#pragma once

#include "core/data.h"
#include "core/model.h"
#include "core/types.h"
#include <Eigen/Dense>

namespace mlfs {
class PolynomialRegression : public Model {
public:
  /**
   * Initialises the model with the initial options
   * @param options The options struct
   */
  PolynomialRegression(int feature_size, PolynomialRegressionOptions options);
  ~PolynomialRegression() = default;

  void fit(mlfs::CSVDataset &data) override;
  Eigen::VectorXd predict(const mlfs::RowMatrixXd &X) const override;

  // Getters
  [[nodiscard]] const Eigen::VectorXd &get_weights() const { return weights_; }
  [[nodiscard]] const LinearRegressionOptions &get_opts() const {
    return opts_;
  }

private:
  Eigen::VectorXd weights_;
  const LinearRegressionOptions opts_;

  /**
   * Finds the optimal weights directly using a closed form approach
   * @param X The features
   * @param Y The target values
   */
  void fit_closed_form(const mlfs::RowMatrixXd &X, const Eigen::VectorXd &Y);

  /**
   * Trains the model on features and targerts using a gradient descent
   * @param X The features
   * @param Y The target values
   */
  void fit_gd(const mlfs::RowMatrixXd &X, const Eigen::VectorXd &Y);
};
} // namespace mlfs
