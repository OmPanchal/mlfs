#include "core/data.h"
#include "core/encoders.h"
#include "core/losses.h"
#include "core/types.h"
#include "regression/linear_regression.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>
#include <unordered_map>

mlfs::RowMatrixXd load_data() {
  mlfs::CSVLoader loader = mlfs::CSVLoader();

  mlfs::RowMatrixXd data = loader.load_csv_to_row_matrix(
      "/home/om/Programming/C++Sandbox/mlfs/tests/test_data.csv");

  return data;
}

TEST(DataTest, CheckLoad) {
  mlfs::RowMatrixXd output(5, 3);
  output << 1., 1., 1., 2., 0., 1., 3., 3., 1., 0., 3., 1., 5., 5., 1.;

  EXPECT_EQ(load_data(), output);
}