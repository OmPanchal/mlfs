#pragma once
#include "core/types.h"
#include <Eigen/Dense>

namespace mlfs {
/**
 * A basic Model Interface
 * */
class Model {
public:
  virtual ~Model() = default;
  /**
   * Trains the model on the features X using the targets y
   * @param X The features
   * @param Y The target values
   */
  virtual void fit(const mlfs::RowMatrixXd &X, const Eigen::VectorXd &Y) = 0;

  /**
   * Makes a prediction based on the feature values provided
   * @param X The features
   * @return The model's predictions
   */
  virtual Eigen::VectorXd predict(const mlfs::RowMatrixXd &X) const = 0;
};
} // namespace mlfs