#include "core/data.h"
#include "core/encoders.h"
#include "core/losses.h"
#include "core/types.h"
#include "regression/linear_regression.h"
#include <Eigen/Dense>
#include <iostream>
#include <unordered_map>

int main() {
  std::unordered_map<std::string, double> cat_map = {
      {"Low", 0.0},
      {"Medium", 2.0},
      {"High", 3.0},
  };

  std::vector<std::string> categories = {"Low", "Medium", "High"};

  mlfs::CSVLoader loader = mlfs::CSVLoader({.separator = ';'});
  loader.add_encoder("col2", std::make_unique<mlfs::OneHotEncoder>(categories));

  mlfs::RowMatrixXd data = loader.load_csv_to_row_matrix(
      "/home/om/Programming/C++Sandbox/mlfs/examples/test.csv");

  std::cout << data << std::endl;
}