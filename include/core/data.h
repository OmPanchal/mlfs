#pragma once

#include "core/encoders.h"
#include "core/types.h"
#include <Eigen/Dense>
#include <iostream>
#include <memory>

namespace mlfs {

void null_to_double(const std::string &p_str, double &p_val,
                    double fallback_value);

/**
 * Wrapper for the output of the result of the `load_csv_to_row_matrix` method
 * in the CSVLoader class. Adds a level of abstraction for the data inputted to
 * the models. And also acts as a scalable interface for any data manipulation
 * methods that could be added in the future
 */
class CSVDataset {
public:
  /** Initialises the wrapper class
   * @param data A `RowMatrixXd` for the data to be stored in the container7
   * @param target_col_idx The zero indexed value of the target column. By
   * default is -1 to represent that the default column will be the last
   * value.
   */
  CSVDataset(RowMatrixXd data, int target_col_idx)
      : target_col_idx_(target_col_idx) {
    features_ = data;
    target_ = extract_column(features_, target_col_idx);
  }

  /**
   * Returns the stored data
   * @return The stored data with the padded ones
   */
  [[nodiscard]] const RowMatrixXd &get_raw_features() const {
    return features_;
  }

  /**
   * Returns the target column as a vector
   * @return The target column
   */
  [[nodiscard]] const Eigen::VectorXd &get_target() const { return target_; }

  /**
   * Returns the stored data
   * @return The stored data without the padded ones
   */
  [[nodiscard]] RowMatrixXd get_features() const {
    return features_.leftCols(features_.cols() - 1);
  }

  friend std::ostream &operator<<(std::ostream &os, const CSVDataset &d) {
    os << d.get_features();
    return os;
  }

private:
  RowMatrixXd features_;
  Eigen::VectorXd target_;
  int target_col_idx_;

  /**
   * Splits the raw data matrix into features and target. The target column is
   * returned
   * @param mat The raw data
   * @param col The column to be extracted
   * @return The extracted column
   */
  static Eigen::VectorXd extract_column(RowMatrixXd &mat, int col);
};

/**
 * The CSV Loader class
 */
class CSVLoader {
public:
  CSVLoader(CSVLoaderOptions options = {}) : opts_(std::move(options)) {}

  /**
   * Adds an encoder into the encoder map. The encoder map stores a mapping of
   * column names and their corresponding encoders.
   * @param col_name The name of the column
   * @param encoder The Encoder object for that column
   */
  void add_encoder(std::string col_name, std::unique_ptr<Encoder> encoder);

  /**
   * Loads a purely numeric CSV file into an Eigen matrix
   *
   * @param filepath The file path to the CSV file
   * @param has_header True if the file has a header for the column names
   * @param separator The separator used in the provided CSV file
   * @return A CSVDataset wrapper containing the contents of the file as a
   * RowMatrixXd
   */
  CSVDataset load_csv_to_row_matrix(const std::string &filepath,
                                    std::string target_col);

private:
  EncoderMap encoder_map_ = {};
  const CSVLoaderOptions opts_;
  std::string target_col;
};
} // namespace mlfs