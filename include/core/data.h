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

/**
 * Loads a purely numeric CSV file into an Eigen matrix
 *
 * @param filepath The file path to the CSV file
 * @param has_header True if the file has a header for the column names
 * @param separator The separator used in the provided CSV file
 * @return A Row Major matrix containing the contents of the file
 */
RowMatrixXd load_csv_to_row_matrix(const std::string &filepath,
                                   bool has_header = true, char separator = ',',
                                   const EncoderMap &encoder_map = {}) {

  rapidcsv::LabelParams labelParams =
      has_header ? rapidcsv::LabelParams(0, -1) : rapidcsv::LabelParams(-1, -1);
  rapidcsv::Document doc(filepath, labelParams,
                         rapidcsv::SeparatorParams(separator, true));

  size_t numCols = doc.GetColumnCount();
  size_t numRows = doc.GetRowCount();

  // Handle empty file
  if (numRows == 0 || numCols == 0) {
    return Eigen::MatrixXd(0, 0);
  }
  // Allocate appropriate space to the Eigen Matrix
  RowMatrixXd matrix(numRows, numCols);

  for (size_t col_idx = 0; col_idx < numCols; col_idx++) {
    // see if the column name exists in the
    std::string col_name = doc.GetColumnName(col_idx);
    auto it = encoder_map.find(col_name);

    if (it != encoder_map.end()) {
      std::vector<std::string> raw_column = doc.GetColumn<std::string>(col_idx);
      int colsWritten = it->second->transform(raw_column, matrix, col_idx);

    } else {
      // ! [NUMERIC ONLY] Parse the column
      std::vector<double> columnData = doc.GetColumn<double>(col_idx);
      // Set the respective Eigen Matrix column into the parsed value.
      matrix.col(col_idx) = Eigen::Map<const Eigen::VectorXd>(
          columnData.data(), columnData.size());
    }
  }

  return matrix;
}
} // namespace mlfs