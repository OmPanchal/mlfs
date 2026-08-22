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

  mlfs::RowMatrixXd data = loader.load_csv_to_row_matrix(
      "/home/om/Programming/C++Sandbox/mlfs/examples/test.csv");

  Eigen::VectorXd targets(11);
  targets << 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.;

  std::cout << data << "\n\n";

  mlfs::LinearRegression model =
      mlfs::LinearRegression(2, {.learning_rate = 0.001,
                                 .epochs = 10000,
                                 .batch_size = 5,
                                 .solver = mlfs::SolverType::GD,
                                 .lambda = 0.5,
                                 .alpha = 0.5});
  model.fit(data, targets);

  std::cout << "W = " << model.get_weights() << "\n";
}