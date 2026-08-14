#include "core/losses.h"

namespace mlfs {
double MSE::compute(Eigen::VectorXd &Y, Eigen::VectorXd &Y_pred) const {
  return (Y - Y_pred).squaredNorm() / Y.size();
}
} // namespace mlfs