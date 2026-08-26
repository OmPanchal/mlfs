#pragma once

#include "core/types.h"
#include <iostream>
#include <unordered_map>
#include <vector>

namespace mlfs {

class Encoder {
public:
  /**
   * The base Encoder Interface
   */
  virtual ~Encoder() = default;

  [[nodiscard]] virtual int
  transform(const std::vector<std::string> &raw_column,
            RowMatrixXd &target_matrix, int column_idx) = 0;

  virtual int get_output_column_count() const { return 1; }
};

class OrdinalEncoder : public Encoder {
public:
  /**
   * The Ordinal Encoder class
   *
   * @param category_map The mapping of the categories to their values
   * @param fallback The default value in case a pair in the mapping do not
   * exist
   */
  OrdinalEncoder(std::unordered_map<std::string, double> category_map,
                 double fallback = -1.);
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
  [[nodiscard]] int transform(const std::vector<std::string> &raw_column,
                              RowMatrixXd &target_matrix,
                              int column_idx) override;

private:
  std::unordered_map<std::string, double> category_map_;
  double fallback_;
};

class OneHotEncoder : public Encoder {
public:
  /**
   * The One Hot Encoder Class
   *
   * @param categories A vector of the different categorires part of the column
   */
  OneHotEncoder(const std::vector<std::string> categories);

  /**
   * Performs one hot encoding on the provided column using `categories`.
   * Updates the target vector direcly.
   *
   * @param raw_column The raw column of strings to be encoded
   * @param target_matrix The target matrix to store the column
   * @param column_idx The index of the column to be encoded
   * @return The number of columns that were modified
   */
  [[nodiscard]] int transform(const std::vector<std::string> &raw_column,
                              RowMatrixXd &target_matrix,
                              int column_idx) override;

  /**
   * Returns the number of outputs columns the encoding modifies
   * @return The number of modified columns (including ones added)
   */
  int get_output_column_count() const override;

private:
  std::unordered_map<std::string, size_t> categories_;
};

using EncoderMap = std::unordered_map<std::string, std::unique_ptr<Encoder>>;
} // namespace mlfs