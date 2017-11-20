#include "fit.hpp"
#include <armadillo>
#include <iostream>
#include <map>
#include <string>

using namespace arma;
void Fit::SetParams(std::string loss_type) { loss_type_ = loss_type; }

///@brief Proximal gradient decent for vector of lambda values.
///@param[in] x
/// Covariate matrix
///@param[in] y
/// Outcome Vector
///@param[in] lambda
/// Vector of lambda values to use
///@return Matrix containing beta values for all lambda values.
mat Fit::l2ggd(mat x, colvec y, colvec lambda) {
  mat results;
  colvec b;
  b = colvec(x.n_cols, 1);
  b.fill(0);
  results = mat(lambda.size(), x.n_cols);
  colvec update = b;

  for (int j = 0; j < lambda.size(); j++) {
    do {
      b = update;
      double t = step_size(b, x, y, lambda(j));
      update = l2ggdupdate(x, y, b, lambda(j), t);
    } while (norm(update - b) > 1e-5);
    results.row(j) = b.t();
  }
  return results;
}
///@brief Proximal gradient decent for a single lambda value.
///@param[in] x
/// Covariate matrix
///@param[in] y
/// Outcome Vector
///@param[in] lambda
/// Single lambda value to use
///@return Vector containing beta values.
colvec Fit::l2ggd(mat x, colvec y, double lambda) {
  colvec b;
  b = colvec(x.n_cols, 1);
  b.fill(0);
  colvec update = b;
  do {
    b = update;
    double t = step_size(b, x, y, lambda);
    update = l2ggdupdate(x, y, b, lambda, t);
  } while (norm(update - b) > 1e-5);
  return b;
}
colvec Fit::l2ggdupdate(mat x, colvec y, colvec b, double lambda, double t) {
  colvec update;
  update = Fit::soft_threshold(b + (t) * (x.t() / x.n_rows) * (y - (x * b)),
                               t * lambda);
  return update;
}

double Fit::step_size(colvec b, mat x, colvec y, double lambda) {
  double t = 2.0;
  double update;
  double oldupdate =
      .5 * norm(y - x * b) * norm(y - x * b) + lambda * norm(b, 1);
  double realoldupdate;
  do {
    t *= .5;
    update = .5 * norm(y - (x * l2ggdupdate(x, y, b, lambda, t))) *
                 norm(y - (x * l2ggdupdate(x, y, b, lambda, t))) +
             lambda * norm(l2ggdupdate(x, y, b, lambda, t), 1);
    realoldupdate = oldupdate -
                    .5 * t *
                        norm((1 / t) * (b - l2ggdupdate(x, y, b, lambda, t))) *
                        norm((1 / t) * (b - l2ggdupdate(x, y, b, lambda, t)));
  } while (update > oldupdate);
  return t;
}
///@brief Soft threshold function
colvec Fit::soft_threshold(colvec x, double lambda) {
  for (int i = 0; i < x.size(); i++) {
    if (x(i) > lambda)
      x(i) = x(i) - lambda;
    else if (x(i) < -lambda)
      x(i) = x(i) + lambda;
    else
      x(i) = 0;
  }
  return x;
}
