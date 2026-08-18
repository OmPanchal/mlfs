#include "core/encoders.h"
#include "core/types.h"
#include <iostream>
#include <unordered_map>
#include <vector>

namespace mlfs {
// Ordinal
OrdinalEncoder::OrdinalEncoder(
    std::unordered_map<std::string, double> category_map, double fallback) {
  category_map_ = category_map;
  fallback_ = fallback;
}

int OrdinalEncoder::transform(const std::vector<std::string> &raw_column,
                              RowMatrixXd &target_matrix, int column_idx) {
  for (size_t row = 0; row < raw_column.size(); row++) {
    // Check if the current item has a map value
    auto it = category_map_.find(raw_column[row]);
    target_matrix(row, column_idx) =
        (it != category_map_.end()) ? it->second : fallback_;
  }
  // One column modified
  return 1;
}

// One Hot
OneHotEncoder::OneHotEncoder(const std::vector<std::string> categories) {
  for (size_t i = 0; i < categories.size(); i++) {
    categories_.insert({categories[i], i});
  }
}

int OneHotEncoder::transform(const std::vector<std::string> &raw_column,
                             RowMatrixXd &target_matrix, int column_idx) {

  for (size_t row = 0; row < raw_column.size(); row++) {
    auto it = categories_.find(raw_column[row]);
    if (it != categories_.end()) {
      target_matrix(row, column_idx + it->second) = 1.;
    } else {
      target_matrix(row, column_idx + categories_.size()) = 1.;
    }
  }
  return categories_.size() + 1;
}

int OneHotEncoder::get_output_column_count() const {
  return categories_.size() + 1;
}

} // namespace mlfs
