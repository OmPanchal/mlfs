#include "core/data.h"
#include "core/encoders.h"
#include "core/losses.h"
#include "core/types.h"
#include "regression/linear_regression.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>
#include <unordered_map>

/**
 * Test only numeric data with no null values
 */
TEST(DataLoadTest, NumericDataSet) {
  mlfs::CSVLoader loader = mlfs::CSVLoader();
  mlfs::RowMatrixXd data = loader.load_csv_to_row_matrix(
      "/home/om/Programming/C++Sandbox/mlfs/data/numeric_data.csv");

  mlfs::RowMatrixXd output{
      {1.23, 2., 1.},       {2., 2.12, 1.}, {2., 3., 1.},
      {123.54, 231.33, 1.}, {5., 5., 1.},
  };

  EXPECT_EQ(data, output);
}

/**
 * Test only numeric data with all the different null values that the load
 * function handles
 */
TEST(DataLoadTest, NullNumericDataSet) {
  mlfs::CSVLoader loader = mlfs::CSVLoader();
  mlfs::RowMatrixXd data = loader.load_csv_to_row_matrix(
      "/home/om/Programming/C++Sandbox/mlfs/data/null_numeric_data.csv");

  mlfs::RowMatrixXd output{
      {1., 0., 1.}, {0., 0., 1.}, {0., 3., 1.}, {0., 0., 1.}, {5., 5., 1.},
  };

  EXPECT_EQ(data, output);
}

/**
 * Test only numeric data with all the different null values that the load
 * function handles with a custom fallback value
 */
TEST(DataLoadTest, NullNumericDataSetWithCustomFallback) {
  mlfs::CSVLoader loader = mlfs::CSVLoader({.null_value_fallback = -4.234});
  mlfs::RowMatrixXd data = loader.load_csv_to_row_matrix(
      "/home/om/Programming/C++Sandbox/mlfs/data/null_numeric_data.csv");

  mlfs::RowMatrixXd output{
      {1., -4.234, 1.},     {-4.234, -4.234, 1.}, {-4.234, 3., 1.},
      {-4.234, -4.234, 1.}, {5., 5., 1.},
  };

  EXPECT_EQ(data, output);
}

/**
 * Test mixed data with no null values and ordinal encoding
 */
TEST(DataLoadTest, MixedDataSetWithOrdinalEncoding) {
  mlfs::CSVLoader loader = mlfs::CSVLoader();

  std::unordered_map<std::string, double> category_map_1;
  category_map_1.insert({"High", 1});
  category_map_1.insert({"Normal", 2.5});
  category_map_1.insert({"Low", -3});

  loader.add_encoder("col2",
                     std::make_unique<mlfs::OrdinalEncoder>(category_map_1));

  mlfs::RowMatrixXd data = loader.load_csv_to_row_matrix(
      "/home/om/Programming/C++Sandbox/mlfs/data/mixed_data.csv");

  mlfs::RowMatrixXd output{
      {1., 1., 5., 1.},   {23., -3, 5., 1.},       {8., 2.5, 5., 1.},
      {-23., 1., 5., 1.}, {-3.14159, -3., 5., 1.},
  };

  EXPECT_EQ(data, output);
}

/**
 * Test mixed data with null values and typo for non numeric data using
 * ordinal encoding
 */
TEST(DataLoadTest, NullMixedDataSetWithOrdinalEncoding) {
  mlfs::CSVLoader loader = mlfs::CSVLoader();

  std::unordered_map<std::string, double> category_map_1;
  category_map_1.insert({"High", 1});
  category_map_1.insert({"Normal", 2.5});
  category_map_1.insert({"Low", -3});

  loader.add_encoder("col2",
                     std::make_unique<mlfs::OrdinalEncoder>(category_map_1));

  mlfs::RowMatrixXd data = loader.load_csv_to_row_matrix(
      "/home/om/Programming/C++Sandbox/mlfs/data/null_mixed_data.csv");

  mlfs::RowMatrixXd output{
      {0., 1., 0., 1.},    {23., -3, 5., 1.},  {0., 2.5, 0., 1.},
      {-23., -1., 0., 1.}, {-0., -3., 5., 1.},
  };

  EXPECT_EQ(data, output);
}

/**
 * Test mixed data with null values and typo for non numeric data using
 * ordinal encoding and a custom fallback value for the encoder
 */
TEST(DataLoadTest, NullMixedDataSetWithOrdinalEncodingWithCustomFallback) {

  mlfs::CSVLoader loader = mlfs::CSVLoader();

  std::unordered_map<std::string, double> category_map_1;
  category_map_1.insert({"High", 1});
  category_map_1.insert({"Normal", 2.5});
  category_map_1.insert({"Low", -3});

  loader.add_encoder(
      "col2", std::make_unique<mlfs::OrdinalEncoder>(category_map_1, 12.2));

  mlfs::RowMatrixXd data = loader.load_csv_to_row_matrix(
      "/home/om/Programming/C++Sandbox/mlfs/data/null_mixed_data.csv");

  mlfs::RowMatrixXd output{
      {0., 1., 0., 1.},     {23., -3, 5., 1.},  {0., 2.5, 0., 1.},
      {-23., 12.2, 0., 1.}, {-0., -3., 5., 1.},
  };

  EXPECT_EQ(data, output);
}

/**
 * Test mixed data with no null values and one hot encoding
 */
TEST(DataLoadTest, MixedDataSetWithOneHotEncoding) {
  mlfs::CSVLoader loader = mlfs::CSVLoader();

  std::vector<std::string> categories = {"Low", "Normal", "High"};

  loader.add_encoder("col2", std::make_unique<mlfs::OneHotEncoder>(categories));

  mlfs::RowMatrixXd data = loader.load_csv_to_row_matrix(
      "/home/om/Programming/C++Sandbox/mlfs/data/mixed_data.csv");

  mlfs::RowMatrixXd output{
      {1., 0., 0., 1., 0., 5., 1.},       {23., 1., 0., 0., 0., 5., 1.},
      {8., 0., 1., 0., 0., 5., 1.},       {-23., 0., 0., 1., 0., 5., 1.},
      {-3.14159, 1., 0., 0., 0., 5., 1.},
  };

  EXPECT_EQ(data, output);
}

/**
 * Test mixed data with null values and typo for non numeric data using
 * one hot encoding
 * ! The One hot encoding does not have a custom fallback so it is not tested
 * ! for as it is with the ordinal encoder
 */
TEST(DataLoadTest, NullMixedDataSetWithOneHotEncoding) {
  mlfs::CSVLoader loader = mlfs::CSVLoader();

  std::vector<std::string> categories = {"Low", "Normal", "High"};

  loader.add_encoder("col2", std::make_unique<mlfs::OneHotEncoder>(categories));

  mlfs::RowMatrixXd data = loader.load_csv_to_row_matrix(
      "/home/om/Programming/C++Sandbox/mlfs/data/null_mixed_data.csv");

  mlfs::RowMatrixXd output{
      {0., 0., 0., 1., 0., 0., 1.}, {23., 1., 0., 0., 0., 5., 1.},
      {0., 0., 1., 0., 0., 0., 1.}, {-23., 0., 0., 0., 1., 0., 1.},
      {0, 1., 0., 0., 0., 5., 1.},
  };

  EXPECT_EQ(data, output);
}
