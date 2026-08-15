#pragma once

#include "core/types.h"
#include <unordered_map>
#include <vector>

namespace mlfs {

class Encoder {
public:
  virtual ~Encoder() = default;

  virtual int transform(const std::vector<std::string> &raw_column,
                        RowMatrixXd &target_matrix, int column_idx) = 0;
};

class OrdinalEncoder : public Encoder {
  /**
   * The base Encoder Interface
   */
public:
  /**
   * The Ordinal Encoder class
   * @param category_map The mapping of the categories to their values
   * @param fallback The default value in case a pair in the mapping do not
   * exist
   */
  OrdinalEncoder(std::unordered_map<std::string, double> category_map,
                 double fallback = 0.0) {
    category_map_ = category_map;
    fallback_ = fallback;
  };
  ~OrdinalEncoder() = default;

  /**
   * Performs ordinal encoding on the provided column using `category_map`.
   * Updates the target vector direcly.
   *
   * @param raw_column The raw column of strings to be encoded
   * @param target_matrix The target matrix to store the column
   * @param column_idx The index of the column to be encoded
   * @return The number of columns that were modified
   */
  int transform(const std::vector<std::string> &raw_column,
                RowMatrixXd &target_matrix, int column_idx) override {
    for (size_t row = 0; row < raw_column.size(); row++) {

      // Check if the current item has a map value
      auto it = category_map_.find(raw_column[row]);
      target_matrix(row, column_idx) =
          (it != category_map_.end()) ? it->second : fallback_;
    }
    // One column modified
    return 1;
  }

private:
  std::unordered_map<std::string, double> category_map_;
  double fallback_;
};

using EncoderMap = std::unordered_map<std::string, std::unique_ptr<Encoder>>;
} // namespace mlfs