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
  CV test = CV(X, Y, 5, "l2");
  test.PartitionCycle();

  double l = test.ReturnBestLambda();
  colvec betahat = test.opt.ggd(test.x_, test.y_, 0.04084348);
  cout << betahat << endl;
  colvec glmbest;
  glmbest.load("best.txt");
  /// Ensure coefficients are similar to glmnet.
  REQUIRE(norm(betahat - glmbest) < .05);
  /// Recovers approximate lambda value from glmnet.
  REQUIRE((l - 0.04084348) * ((l - 0.04084348)) < .001);
  cout << l << endl;
}
TEST_CASE("log", "[log]") {
  /// Initialize a column vector Y from our data. Test data has n=30, p=40, s=2,
  /// seed=1408.
  /// See datagen.r for more info.
  colvec Y;
  Y.load("Ylog.txt");
  /// Initialize a matrix X from our data.
  mat X;
  X.load("Xlog.txt");
  /// Create GGD object.
  CV test = CV(X, Y, 10, "log");
  double l = test.ReturnBestLambda();
  colvec betahat = test.opt.ggd(test.x_, test.y_, 0.01882035);
  colvec glmbetahat;
  glmbetahat.load("logbest.txt");
  /// Ensure coefficients are similar to glmnet.
  REQUIRE(norm(betahat - glmbetahat) < .01);
}
}
