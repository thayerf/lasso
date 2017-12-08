#ifndef LASSO_OPTIMIZE_
#define LASSO_OPTIMIZE_
#include <armadillo>
#include <iostream>
#include <map>
#include <string>

using namespace arma;
class Fit {
 private:
 public:
  Fit(){};
  std::string loss_type_;
  void SetParams(std::string loss_type);
  mat ggd(mat x, colvec y, colvec lambda);
  colvec ggd(mat x, colvec y, double lambda);
  colvec soft_threshold(colvec x, double lambda);
  colvec fit(mat x, colvec y, colvec b, double lambda);
  colvec l2ggdupdate(mat x, colvec y, colvec b, double lambda, double t);
  colvec l2grad(mat x, colvec y, colvec b);
  double l2loss(mat x, colvec y, colvec b);
  colvec loggrad(mat x, colvec y, colvec b);
  double logloss(mat x, colvec y, colvec b);
};
#endif  // LASSO_OPTIMIZE
