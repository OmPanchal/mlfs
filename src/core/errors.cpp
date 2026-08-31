#include "core/errors.h"

namespace mlfs {
std::string REGRESSION_NO_SOLUTION =
    "[Error] A closed form solution does not exist for the model with alpha > "
    "0 (i.e. a model with l1 regularisation). Please try using gradient "
    "descent instead.";

std::string MATRIX_NOT_INVERTIBLE =
    "[Error] Closed form solution not possible due to "
    "non invertable matrix. Please try using gradient descent instead.";

std::string NON_POSITIVE_LEARNING_RATE =
    "[Invalid Argument] learning_rate must be positive.";

std::string NON_POSITIVE_EPOCHS = "[Invalid Argument] epochs must be positive.";

std::string NON_POSITIVE_BATCH_SIZE =
    "[Invalid Argument] batch_size must be positive.";

std::string REGRESSION_NEGATIVE_LAMBDA =
    "[Invalid Argument] lambda must be positive.";

std::string REGRESSION_OUT_OF_RANGE_ALPHA =
    "[Invalid Argument] alpha must be in the interval [0, 1].";

std::string POLYNOMIAL_REGRESSION_LINEAR_ORDER =
    "[Invalid Argument] order must be greater than two for this polynomial "
    "regression model. Make use of Linear Regression if you wish for a "
    "polynomial regression model with order=1.";

std::string POLYNOMIAL_REGRESSION_NON_POSITIVE_ORDER =
    "[Invalid Argument] order must be positive.";

std::string TARGET_COL_IDX_NOT_SET =
    "[Invalid Argument] The target column has not been set, make sure that the "
    "target column name has been type correctly.";

std::string SOLVER_DOES_NOT_EXIST =
    "[Invalid Argument] This solver does not exist.";
} // namespace mlfs
