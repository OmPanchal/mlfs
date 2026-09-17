#pragma once

#include "core/model.h"
#include "core/regularisers.h"
#include "core/types.h"
#include <Eigen/Dense>

namespace mlfs {
class LogisticRegression : public Model {
public:
  /**
   * Initialises the model with the initial options
   * @param feature_size The feature size of the model input
   * @param options The option struct
   */
  LogisticRegression(int feature_size, RegressionOptions options);
  ~LogisticRegression() = default;

  void fit(mlfs::CSVDataset &dataset) override;
  Eigen::VectorXd predict(const mlfs::RowMatrixXd &X) const override;

  // Getters
  [[nodiscard]] const Eigen::VectorXd &get_weights() const { return weights_; }
  [[nodiscard]] const double get_bias() const { return bias_; }
  [[nodiscard]] const RegressionOptions &get_opts() const { return opts_; }

  static inline Eigen::VectorXd sigmoid(Eigen::VectorXd X) {
    return (1 / (1 + X.array().exp())).matrix();
  }

private:
  Eigen::VectorXd weights_;
  double bias_;
  const RegressionOptions opts_;
  const std::unique_ptr<NormRegulariser> l1_regulariser =
      std::make_unique<L1Regulariser>();
  const std::unique_ptr<NormRegulariser> l2_regulariser =
      std::make_unique<L2Regulariser>();

  /**
   * Trains the model on features and targerts using a gradient descent
   * @param X The features
   * @param Y The target values
   */
  void fit_gd(const mlfs::RowMatrixXd &X, const Eigen::VectorXd &Y);
};
} // namespace mlfs
