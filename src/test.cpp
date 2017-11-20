// This tells Catch to provide a main() - only do this in one cpp file.
#define CATCH_CONFIG_MAIN
#include <armadillo>
#include <iostream>
#include "catch.hpp"
#include "cv.hpp"
#include "fit.hpp"
namespace arma {
TEST_CASE("recovery", "[recorvery]") {
  /// Initialize a column vector Y from our data. Test data has n=30, p=40, s=2,
  /// seed=1408.
  /// See datagen.r for more info.
  colvec Y;
  Y.load("Y.txt");
  /// Initialize a matrix X from our data.
  mat X;
  X.load("X.txt");
  /// Create GGD object.
  CV test = CV(X, Y, 5);
  test.PartitionCycle();

  /// Ensure recoverable beta value is nonzero.
  double l = test.ReturnBestLambda();
  colvec betahat = test.opt.l2ggd(test.x_, test.y_, l);
  REQUIRE(betahat(1) > .1);
  /// Recovers approximate lambda value from glmnet.
  REQUIRE(l - 0.4112781 < .1);
  cout << l << endl;
}
}
