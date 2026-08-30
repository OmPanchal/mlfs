#include "core/data.h"
#include "core/encoders.h"
#include "core/losses.h"
#include "core/types.h"
#include "regression/linear_regression.h"
#include "regression/polynomial_regression.h"
#include <Eigen/Dense>
#include <iostream>
#include <unordered_map>

int main() {
  mlfs::CSVLoader loader = mlfs::CSVLoader();
  mlfs::CSVDataset data = loader.load_csv_to_row_matrix(
      "/home/om/Programming/C++Sandbox/mlfs/data/numeric_data_2.csv", "col2");

  mlfs::RowMatrixXd features = data.get_features();
  std::cout << mlfs::create_vandermonde_matrix(features, 5);
  // mlfs::LinearRegression model =
  //     mlfs::LinearRegression(2, {.learning_rate = 0.0001,
  //                                .epochs = 100000,
  //                                .batch_size = 7,
  //                                .solver = mlfs::SolverType::GD,
  //                                .lambda = 0,
  //                                .alpha = 0});

  // // fit the model on the dataset
  // model.fit(data);

  // std::cout << model.get_weights();
}