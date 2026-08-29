#include "core/data.h"
#include "core/encoders.h"
#include "core/types.h"
#include <Eigen/Dense>
#include <memory>
#include <rapidcsv.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlfs {
Eigen::VectorXd CSVDataset::extract_column(RowMatrixXd &mat, int col) {
  // Get the extracted column
  Eigen::VectorXd extracted_col = mat.col(col);

  Eigen::Index num_cols_to_move = mat.cols() - col - 1;
  if (num_cols_to_move > 0) {
    // Shift the right most columns to the column to be extracted one space left
    // Eval used for safety
    mat.middleCols(col, num_cols_to_move) =
        mat.rightCols(num_cols_to_move).eval();
  }

  // Resize the matrix as it has shrunk
  mat.conservativeResize(mat.rows(), mat.cols() - 1);
  return extracted_col;
}

void null_to_double(const std::string &p_str, double &p_val,
                    double fallback_value) {
  // Added some standard "null" placeholders for now...
  if (p_str.empty() || p_str == "n/a" || p_str == "N/A" || p_str == "na" ||
      p_str == "NA" || p_str == " ") {
    p_val = fallback_value;
    return;
  }
  // Set the value to the converted double value
  p_val = std::strtod(p_str.c_str(), nullptr);
}

void CSVLoader::add_encoder(std::string col_name,
                            std::unique_ptr<Encoder> encoder) {
  encoder_map_[col_name] = std::move(encoder);
}

CSVDataset CSVLoader::load_csv_to_row_matrix(const std::string &filepath,
                                             std::string target_col) {

  rapidcsv::LabelParams labelParams = opts_.has_header
                                          ? rapidcsv::LabelParams(0, -1)
                                          : rapidcsv::LabelParams(-1, -1);
  rapidcsv::Document doc(filepath, labelParams,
                         rapidcsv::SeparatorParams(opts_.separator, true));

  size_t num_cols = doc.GetColumnCount();
  size_t num_rows = doc.GetRowCount();
  size_t matrix_cols = num_cols - encoder_map_.size();

  // Calculate the maximum output columns
  for (auto &[key, value] : encoder_map_) {
    matrix_cols += value->get_output_column_count();
  }

  // Handle empty file
  if (num_rows == 0 || num_cols == 0) {
    RowMatrixXd empty_matrix(0, 0);
    CSVDataset output(empty_matrix, 0);
    return output;
  }
  // Allocate appropriate space to the Eigen Matrix
  RowMatrixXd matrix = RowMatrixXd::Zero(num_rows, matrix_cols + 1);

  // initial value for the target_col_idx
  int target_col_idx = -1;

  size_t matrix_col_idx = 0;
  for (size_t doc_col_idx = 0; doc_col_idx < num_cols; doc_col_idx++) {
    // see if the column name exists in the
    std::string col_name = doc.GetColumnName(doc_col_idx);
    auto it = encoder_map_.find(col_name);

    // Check if the target column matches the current column
    if (col_name == target_col) {
      target_col_idx = matrix_col_idx;
    }

    // Check for any column encoders
    if (it != encoder_map_.end()) {
      // Apply the encoder's method to the current column
      std::vector<std::string> raw_column =
          doc.GetColumn<std::string>(doc_col_idx);
      int cols_written =
          it->second->transform(raw_column, matrix, matrix_col_idx);

      matrix_col_idx += cols_written;
    } else {

      // String to double converter replaces null values in numeric cells
      const double &fallback = opts_.null_value_fallback;
      auto lambda_null_converter = [fallback](const std::string &p_str,
                                              double &p_val) {
        null_to_double(p_str, p_val, fallback);
      };

      // Otherwise parse numeric columns
      std::vector<double> columnData =
          doc.GetColumn<double>(doc_col_idx, lambda_null_converter);
      // Set the respective Eigen Matrix column into the parsed value.
      matrix.col(matrix_col_idx) = Eigen::Map<const Eigen::VectorXd>(
          columnData.data(), columnData.size());
      matrix_col_idx++;
    }
  }
  // Add a matrix column of ones to avoid copying for the bias
  matrix.rightCols(1).setConstant(1.);

  // Make sure that target_col_idx has been set
  if (target_col_idx == -1) {
    throw std::invalid_argument(TARGET_COL_IDX_NOT_SET);
  }

  // Return the loaded data as a CSVDataset
  CSVDataset output(matrix, target_col_idx);
  return output;
}

} // namespace mlfs