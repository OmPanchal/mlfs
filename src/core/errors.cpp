#include "core/errors.h"

namespace mlfs {
std::string LINEAR_REGRESSION_NO_SOLUTION =
    "[Error] A closed form solution does not exist for the model with alpha > "
    "0 (i.e. a model with l1 regularisation). Please try using gradient "
    "descent instead.";

std::string MATRIX_NOT_INVERTIBLE =
    "[Error] Closed form solution not possible due to "
    "non invertable matrix. Please try using gradient descent instead.";

std::string NEGATIVE_LEARNING_RATE =
    "[Invalid Argument] learning_rate must be positive";

std::string NEGATIVE_EPOCHS = "[Invalid Argument] epochs must be positive";

std::string NEGATIVE_BATCH_SIZE =
    "[Invalid Argument] batch_size must be positive";

std::string LINEAR_REGRESSION_NEGATIVE_LAMBDA =
    "[Invalid Argument] lambda must be positive";

std::string LINEAR_REGRESSION_OUT_OF_RANGE_ALPHA =
    "[Invalid Argument] alpha must be in the interval [0, 1]";
} // namespace mlfs
