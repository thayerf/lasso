#ifndef LASSO_OPTIMIZE_
#define LASSO_OPTIMIZE_
#include <armadillo>
#include <iostream>
#include <map>
#include <string>

using namespace arma;
class Optimize{
private:

public:
  Optimize(){};
  colvec lambda_;
  colvec b_;
  std::string loss_type_;
  void SetParams(colvec lambda, std::string loss_type, int num_params);
  colvec partition_step(mat training_data, colvec training_outcomes,
                        mat testing_data, colvec testing_outcomes);
  double l2loss(mat training_data, colvec training_outcomes,
                mat testing_data, colvec testing_outcomes, colvec coeffs, double lambda);
  colvec l2ggd(mat x, colvec y, colvec b, double lambda);
  double soft_threshold(double x, double lambda);
};
#endif //LASSO_OPTIMIZE
