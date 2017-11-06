// This tells Catch to provide a main() - only do this in one cpp file.
#define CATCH_CONFIG_MAIN
#include <armadillo>
#include <iostream>
#include "catch.hpp"
#include "partition.hpp"
namespace arma {
TEST_CASE("test", "[test]") {
  /// Initialize a column vector Y from our data.
  colvec Y;
  Y.load("Y.txt");
  /// Initialize a matrix X from our data.
  mat X;
  X.load("X.txt");
  /// Create GGD object.
  Partition test = Partition(X,Y,5);
  colvec beta=test.PartitionCycle();
  cout<<beta(0)<<endl;
  cout<<beta(1)<<endl;
  cout<<beta(2)<<endl;
  std::setprecision(3);
  beta.save("betahat.txt",arma_ascii);
}
}
