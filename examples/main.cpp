#include "core/data.h"
#include "core/losses.h"
#include "core/types.h"
#include "regression/linear_regression.h"
#include <Eigen/Dense>
#include <iostream>
#include <unordered_map>

int main() {
  std::unordered_map<std::string, double> col_map = {
      {"Low", 0.0},
      {"Medium", 2.0},
      {"High", 3.0},
  };

  mlfs::EncoderMap encoder_map = {
      {"col2", std::make_unique<mlfs::OrdinalEncoder>(col_map)}};

  mlfs::RowMatrixXd data = mlfs::load_csv_to_row_matrix(
      "/home/om/Programming/C++Sandbox/mlfs/examples/test.csv", true);

  std::cout << data << std::endl;
}