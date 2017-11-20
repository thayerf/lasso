#ifndef LASSO_OPTIMIZE_
#define LASSO_OPTIMIZE_
#include <armadillo>
#include <iostream>
#include <map>
#include <string>

using namespace arma;
class Fit{
private:

public:
  Fit(){};
  std::string loss_type_;
  void SetParams(std::string loss_type);
  mat l2ggd(mat x, colvec y, colvec b, colvec lambda);
  colvec l2ggd(mat x, colvec y,colvec b, double lambda);
  mat l2ggd(mat x, colvec y, colvec lambda);
  colvec l2ggd(mat x, colvec y, double lambda);
  double soft_threshold(double x, double lambda);
};
#endif //LASSO_OPTIMIZE
