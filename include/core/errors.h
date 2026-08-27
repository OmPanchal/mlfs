#pragma once
#include <expected>
#include <string>

namespace mlfs {
// Error Messages
extern std::string REGRESSION_NO_SOLUTION;

extern std::string MATRIX_NOT_INVERTIBLE;

extern std::string NON_POSITIVE_LEARNING_RATE;

extern std::string NON_POSITIVE_EPOCHS;

extern std::string NON_POSITIVE_BATCH_SIZE;

extern std::string REGRESSION_NEGATIVE_LAMBDA;

extern std::string REGRESSION_OUT_OF_RANGE_ALPHA;

extern std::string POLYNOMIAL_REGRESSION_LINEAR_ORDER;

extern std::string POLYNOMIAL_REGRESSION_NON_POSITIVE_ORDER;
} // namespace mlfs