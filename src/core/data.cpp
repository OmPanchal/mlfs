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

void CSVLoader::add_encoder(std::string col_name,
                            std::unique_ptr<Encoder> encoder) {
  encoder_map_[col_name] = std::move(encoder);
}

RowMatrixXd CSVLoader::load_csv_to_row_matrix(const std::string &filepath) {
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
    return Eigen::MatrixXd(0, 0);
  }
  // Allocate appropriate space to the Eigen Matrix
  RowMatrixXd matrix = RowMatrixXd::Zero(num_rows, matrix_cols);

  size_t matrix_col_idx = 0;
  for (size_t doc_col_idx = 0; doc_col_idx < num_cols; doc_col_idx++) {
    // see if the column name exists in the
    std::string col_name = doc.GetColumnName(doc_col_idx);
    auto it = encoder_map_.find(col_name);

    // Check for any column encoders
    if (it != encoder_map_.end()) {
      // Apply the encoder's method to the current column
      std::vector<std::string> raw_column =
          doc.GetColumn<std::string>(doc_col_idx);
      int cols_written =
          it->second->transform(raw_column, matrix, matrix_col_idx);

      matrix_col_idx += cols_written;
    } else {
      // Otherwise parse numeric columns
      std::vector<double> columnData = doc.GetColumn<double>(doc_col_idx);
      // Set the respective Eigen Matrix column into the parsed value.
      matrix.col(matrix_col_idx) = Eigen::Map<const Eigen::VectorXd>(
          columnData.data(), columnData.size());
      matrix_col_idx++;
    }
  }

  return matrix;
}

} // namespace mlfs