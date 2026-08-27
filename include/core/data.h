#pragma once

#include "core/encoders.h"
#include "core/types.h"
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
class Dataset {
public:
  /** Initialises the wrapper class
   * @param data A `RowMatrixXd` for the data to be stored in the container
   */
  Dataset(RowMatrixXd &data) : data_(data) {}

  /**
   * Returns the stored data
   * @return Returns the stored data without the padded ones
   */
  [[nodiscard]] const RowMatrixXd &get_data() const { return data_; }

  /**
   * Returns the stored data
   * @return Returns the stored data with the padded ones
   */
  [[nodiscard]] RowMatrixXd get_raw_data() const {
    return data_.leftCols(data_.cols() - 1);
  }
  /**
   * ? Could possibly add data processing methods here
   */
  friend std::ostream &operator<<(std::ostream &os, const Dataset &d) {
    os << d.get_raw_data();
    return os;
  }

private:
  RowMatrixXd data_;
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
   * @return A Row Major matrix containing the contents of the file
   */
  RowMatrixXd load_csv_to_row_matrix(const std::string &filepath);

private:
  EncoderMap encoder_map_ = {};
  const CSVLoaderOptions opts_;
};
} // namespace mlfs