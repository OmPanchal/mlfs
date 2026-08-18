#include "core/data.h"
#include "core/encoders.h"
#include "core/losses.h"
#include "core/types.h"
#include "regression/linear_regression.h"
#include <Eigen/Dense>
#include <iostream>
#include <unordered_map>

int main() {
  mlfs::CSVLoader loader = mlfs::CSVLoader({
      .separator = ';',
  });

  mlfs::RowMatrixXd data = loader.load_csv_to_row_matrix(
      "/home/om/Programming/C++Sandbox/mlfs/examples/test.csv");

  Eigen::VectorXd targets(11);
  targets << 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.;

  mlfs::LinearRegression model =
      mlfs::LinearRegression(1, {
                                    .learning_rate = 0.01,
                                    .epochs = 1000,
                                    .batch_size = 5,
                                });
  model.fit(data, targets);
}