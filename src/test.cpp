// This tells Catch to provide a main() - only do this in one cpp file.
#define CATCH_CONFIG_MAIN
#include <armadillo>
#include <iostream>
#include "catch.hpp"
#include "ggd.cpp"
namespace arma {
TEST_CASE("test", "[test]") {
  /// Initialize a column vector Y from our data.
  colvec Y;
  Y.load("Y.txt");
  /// Initialize a matrix X from our data.
  mat X;
  X.load("X.txt");
  /// Choose arbitrary step size.
  double t = .1;
  /// Initial guess of 1 for all beta's
  colvec beta(1000, 1);
  beta.fill(1);
  /// Initialize 1 (for now) lambda value.
  colvec lambda(1, 1);
  lambda.fill(1);
  /// Create GGD object.
  GGD test = GGD(X, Y, beta, lambda, t);
  /// Test Lmax Calculation.
  cout << test.CalcLmax(test.x_, test.y_) << endl;
  /// Test single lasso iteration.
  test = test.lasso(test);
  cout<<test.b_[0]<<","<<test.b_[1]<<","<<test.b_[2]<<endl;
}
}
