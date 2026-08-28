#include "core/data.h"
#include "core/encoders.h"
#include "core/losses.h"
#include "core/types.h"
#include "regression/linear_regression.h"
#include <Eigen/Dense>
#include <iostream>
#include <unordered_map>

int main() {
  mlfs::CSVLoader loader = mlfs::CSVLoader();
  mlfs::CSVDataset data = loader.load_csv_to_row_matrix(
      "/home/om/Programming/C++Sandbox/mlfs/data/numeric_data.csv", "col1");

  std::cout << data.get_features() << "\n\n";
  std::cout << data.get_raw_features() << "\n\n";
  std::cout << data.get_target() << "\n\n";
}