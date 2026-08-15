#include "core/data.h"
#include "core/losses.h"
#include "core/types.h"
#include "regression/linear_regression.h"
#include <Eigen/Dense>
#include <iostream>

int main() {
  Eigen::MatrixXd data = mlfs::LoadCSVToEigen(
      "/home/om/Programming/C++Sandbox/mlfs/examples/test.csv", true);

  int batchSize = 32;
  int totalRows = data.rows();

  for (int startIdx = 0; startIdx < totalRows; startIdx += batchSize) {
    int currentBatchSize = std::min(batchSize, totalRows - startIdx);

    // O(1) Zero-copy slice of the dataset
    mlfs::RowMatrixXd batch = data.middleRows(startIdx, currentBatchSize);

    // Example: Inspect batch dimensions or pass directly to C++ ML APIs
    std::cout << "Processing Batch [" << startIdx << ".."
              << (startIdx + currentBatchSize - 1) << "] | " << "Shape: ("
              << batch.rows() << "x" << batch.cols() << ") | " << std::endl;

    // model.forward(batchDataPtr, currentBatchSize, dataset.cols());
  }
}