// This tells Catch to provide a main() - only do this in one cpp file.
#define CATCH_CONFIG_MAIN
#include <iostream>
#include <armadillo>
#include "ggd.cpp"
#include "catch.hpp"
namespace arma{
TEST_CASE("test", "[test]") {
  colvec Y;
  Y.load("Y.txt");
  mat X;
  X.load("X.txt");
  double t= .5;
  colvec beta(1000,1);
  beta.fill(1);
  colvec lambda(1,1);
  lambda.fill(1);
  GGD test= GGD(X,Y,beta,lambda,t);
  cout<<test.CalcLmax(test.x_,test.y_)<<endl;
  //test= test.lasso(test);

}
}

