#pragma once

#include "core/data.h"
#include "core/model.h"
#include "core/types.h"
#include <Eigen/Dense>

namespace mlfs {
/**
 * Creates a Vandermonde matrix given a vector of values
 * @param x The generating vector
 * @param degree The degree of the final column in the matrix
 */
RowMatrixXd create_vandermonde_matrix(RowMatrixXd x, int degree);

class PolynomialRegression : public Model {
public:
  /**
   * Initialises the model with the initial options
   * @param feature_size The feature size of the model input
   * @param options The options struct
   */
  PolynomialRegression(int degree, PolynomialRegressionOptions options);
  ~PolynomialRegression() = default;

  void fit(mlfs::CSVDataset &data) override;
  Eigen::VectorXd predict(const mlfs::RowMatrixXd &X) const override;

  // Getters
  [[nodiscard]] const Eigen::VectorXd &get_weights() const { return weights_; }
  [[nodiscard]] const PolynomialRegressionOptions &get_opts() const {
    return opts_;
  }

private:
  Eigen::VectorXd weights_;
  const int degree_;
  const PolynomialRegressionOptions opts_;

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

  /**
   * Makes a prediction based on the feature values provided taking the
   * Vandermonde matrix as a the input
   * @param X The features
   * @return The model's predictions
   */
  Eigen::VectorXd predict_(const mlfs::RowMatrixXd &X) const;
};
} // namespace mlfs
