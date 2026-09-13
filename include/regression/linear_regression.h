#pragma once

#include "core/model.h"
#include "core/regularisers.h"
#include "core/types.h"
#include <Eigen/Dense>

namespace mlfs {
class LinearRegression : public Model {
public:
  /**
   * Initialises the model with the initial options
   * @param feature_size The feature size of the model input
   * @param options The option struct
   */
  LinearRegression(int feature_size, LinearRegressionOptions options);
  ~LinearRegression() = default;

  void fit(mlfs::CSVDataset &dataset) override;
  Eigen::VectorXd predict(const mlfs::RowMatrixXd &X) const override;

  // Getters
  [[nodiscard]] const Eigen::VectorXd &get_weights() const { return weights_; }
  [[nodiscard]] const LinearRegressionOptions &get_opts() const {
    return opts_;
  }
  [[nodiscard]] const double get_bias() const { return bias_; }

private:
  Eigen::VectorXd weights_;
  double bias_;
  const LinearRegressionOptions opts_;
  const std::unique_ptr<NormRegulariser> l1_regulariser =
      std::make_unique<L1Regulariser>();
  const std::unique_ptr<NormRegulariser> l2_regulariser =
      std::make_unique<L2Regulariser>();

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
