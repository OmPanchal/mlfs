#pragma once

#include "core/encoders.h"
#include "core/types.h"
#include <Eigen/Dense>
#include <memory>
#include <rapidcsv.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlfs {

class CSVLoader {
public:
  CSVLoader(CSVLoaderOptions options = {}) : opts_(std::move(options)) {}

  /**
   * Adds an encoder into the encoder map. The encoder map stores a mapping of
   * column names and their corresponding encoders.
   * @param col_name The name of the column
   * @param encoder The Encoder object for that column
   */
  void add_encoder(std::string col_name, std::unique_ptr<Encoder> encoder) {
    encoder_map_[col_name] = std::move(encoder);
  }

  /**
   * Loads a purely numeric CSV file into an Eigen matrix
   *
   * @param filepath The file path to the CSV file
   * @param has_header True if the file has a header for the column names
   * @param separator The separator used in the provided CSV file
   * @return A Row Major matrix containing the contents of the file
   */
  RowMatrixXd load_csv_to_row_matrix(const std::string &filepath) {

    rapidcsv::LabelParams labelParams = opts_.has_header
                                            ? rapidcsv::LabelParams(0, -1)
                                            : rapidcsv::LabelParams(-1, -1);
    rapidcsv::Document doc(filepath, labelParams,
                           rapidcsv::SeparatorParams(opts_.separator, true));

    size_t num_cols = doc.GetColumnCount();
    size_t num_rows = doc.GetRowCount();
    size_t matrix_cols = num_cols - encoder_map_.size();

    for (auto &[key, value] : encoder_map_) {
      matrix_cols += value->get_output_column_count();
    }

    // Handle empty file
    if (num_rows == 0 || num_cols == 0) {
      return Eigen::MatrixXd(0, 0);
    }
    // Allocate appropriate space to the Eigen Matrix
    RowMatrixXd matrix = RowMatrixXd::Zero(num_rows, matrix_cols);

    size_t matrix_col_idx = 0;
    for (size_t doc_col_idx = 0; doc_col_idx < num_cols; doc_col_idx++) {
      // see if the column name exists in the
      std::string col_name = doc.GetColumnName(doc_col_idx);
      auto it = encoder_map_.find(col_name);

      if (it != encoder_map_.end()) {
        std::vector<std::string> raw_column =
            doc.GetColumn<std::string>(doc_col_idx);
        int cols_written =
            it->second->transform(raw_column, matrix, matrix_col_idx);

        matrix_col_idx += cols_written;
      } else {
        // Parse Numeric Columns
        std::vector<double> columnData = doc.GetColumn<double>(doc_col_idx);
        // Set the respective Eigen Matrix column into the parsed value.
        matrix.col(matrix_col_idx) = Eigen::Map<const Eigen::VectorXd>(
            columnData.data(), columnData.size());
        matrix_col_idx++;
      }
    }

    return matrix;
  }

private:
  EncoderMap encoder_map_ = {};
  const CSVLoaderOptions opts_;
};
} // namespace mlfs