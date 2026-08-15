#pragma once

#include <Eigen/Dense>
#include <memory>
#include <rapidcsv.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlfs {
using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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
                                   EncoderMap encoder_map = {}) {

  rapidcsv::LabelParams labelParams =
      has_header ? rapidcsv::LabelParams(0, -1) : rapidcsv::LabelParams(-1, -1);
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

  for (size_t col_idx = 0; col_idx < numCols; col_idx++) {
    // see if the column name exists in the
    std::string col_name = doc.GetColumnName(col_idx);
    auto it = encoder_map.find(col_name);

    if (it != encoder_map.end()) {
      std::vector<std::string> raw_column = doc.GetColumn<std::string>(col_idx);
      // ! col_idx passed for now
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