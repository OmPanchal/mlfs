#include "core/data.h"
#include "core/encoders.h"
#include "core/losses.h"
#include "core/types.h"
#include "regression/linear_regression.h"
#include "regression/logistic_regression.h"
#include "regression/polynomial_regression.h"
#include <Eigen/Dense>
#include <iostream>
#include <unordered_map>

int main() {
  mlfs::CSVLoader loader = mlfs::CSVLoader();
  mlfs::CSVDataset data = loader.load_csv_to_row_matrix(
      "/home/om/Programming/C++Sandbox/mlfs/data/numeric_data_2.csv", "col2");

  // std::cout << data;

  mlfs::LinearRegression model =
      mlfs::LinearRegression(1, {.learning_rate = 0.01,
                                 .epochs = 1000,
                                 .batch_size = 1,
                                 .solver = mlfs::SolverType::GD,
                                 .lambda = 0,
                                 .alpha = 0});

  // fit the model on the dataset
  model.fit(data);

  std::cout << "\n"
            << model.get_weights() << " " << model.get_bias() << std::endl;
}