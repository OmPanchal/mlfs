#pragma once

#include "core/encoders.h"
#include "core/types.h"
#include <memory>

namespace mlfs {

void null_to_double(const std::string &p_str, double &p_val,
                    double fallback_value);

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