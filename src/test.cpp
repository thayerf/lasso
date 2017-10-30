// This tells Catch to provide a main() - only do this in one cpp file.
#define CATCH_CONFIG_MAIN
#include <armadillo>
#include <iostream>
#include "catch.hpp"
#include "partition.cpp"
namespace arma {
TEST_CASE("test", "[test]") {
  /// Initialize a column vector Y from our data.
  colvec Y;
  Y.load("Y.txt");
  /// Initialize a matrix X from our data.
  mat X;
  X.load("X.txt");
  /// Choose arbitrary step size.
  double t = .05;
  /// Initial guess of 1 for all beta's
  colvec beta(1000, 1);
  beta.fill(0);
  /// Initialize 1 (for now) lambda value.
  colvec lambda(1, 1);
  lambda.fill(.05);
  /// Create GGD object.
  Partition test = Partition(X,Y,beta,t,2);
  beta=test.PartitionCycle();
  cout<<beta(0)<<endl;
  cout<<beta(1)<<endl;
  cout<<beta(2)<<endl;
  std::setprecision(3);
  beta.save("betahat.txt",arma_ascii);
}
}
