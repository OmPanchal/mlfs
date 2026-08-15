#pragma once

#include <Eigen/Dense>
#include <rapidcsv.h>
#include <string>
#include <vector>

namespace mlfs {
using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
/**
 * Loads a purely numeric CSV file into an Eigen matrix
 * TODO: Try to add ways to handle non-numerical values
 *
 * @param filepath The file path to the CSV file
 * @param hasHeader True if the file has a header for the column names
 * @param separator The separator used in the provided CSV file
 * @return A Row Major matrix containing the contents of the file
 */
RowMatrixXd LoadCSVToEigen(const std::string &filepath, bool hasHeader = true,
                           char separator = ',') {
  rapidcsv::LabelParams labelParams =
      hasHeader ? rapidcsv::LabelParams(0, -1) : rapidcsv::LabelParams(-1, -1);
  rapidcsv::Document doc(filepath, labelParams,
                         rapidcsv::SeparatorParams(separator));

  size_t numCols = doc.GetColumnCount();
  size_t numRows = doc.GetRowCount();

  // Handle empty file
  if (numRows == 0 || numCols == 0) {
    return Eigen::MatrixXd(0, 0);
  }
  // Allocate appropriate space to the Eigen Matrix
  RowMatrixXd matrix(numRows, numCols);

  for (size_t colIdx = 0; colIdx < numCols; colIdx++) {
    // Parse the column
    std::vector<double> columnData = doc.GetColumn<double>(colIdx);
    // Set the respective Eigen Matrix column into the parsed value.
    matrix.col(colIdx) =
        Eigen::Map<const Eigen::VectorXd>(columnData.data(), columnData.size());
  }

  return matrix;
}
} // namespace mlfs