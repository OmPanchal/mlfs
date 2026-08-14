#include "core/losses.h"
#include "core/types.h"
#include "regression/linear_regression.h"
#include <Eigen/Dense>
#include <iostream>

using namespace std;

int main() {
  mlfs::LinearRegression default_model(5, {.learning_rate = 0.0003});

  cout << default_model.getWeights() << "\n\n";
  cout << default_model.getBias() << "\n\n";
  cout << default_model.getOpts().learning_rate << "\n\n";
  cout << default_model.getOpts().batch_size;
}