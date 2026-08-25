#pragma once
#include <expected>
#include <string>

namespace mlfs {
// Error Messages
extern std::string LINEAR_REGRESSION_NO_SOLUTION;

extern std::string MATRIX_NOT_INVERTIBLE;

extern std::string NEGATIVE_LEARNING_RATE;

extern std::string NEGATIVE_EPOCHS;

extern std::string NEGATIVE_BATCH_SIZE;

extern std::string LINEAR_REGRESSION_NEGATIVE_LAMBDA;

extern std::string LINEAR_REGRESSION_OUT_OF_RANGE_ALPHA;
} // namespace mlfs