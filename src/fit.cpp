#include "fit.hpp"
#include <math.h>
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
mat Fit::ggd(mat x, colvec y, colvec lambda) {
  mat results;
  colvec b;
  b = colvec(x.n_cols, 1);
  b.fill(0);
  results = mat(lambda.size(), x.n_cols);
  colvec update = b;

  for (int j = 0; j < lambda.size(); j++) {
    do {
      b = update;
      update = fit(x, y, b, lambda(j));
    } while (norm(update - b) > 1e-10);
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
colvec Fit::ggd(mat x, colvec y, double lambda) {
  colvec b;
  b = colvec(x.n_cols, 1);
  b.fill(0);
  colvec update = b;
  do {
    b = update;
    update = fit(x, y, b, lambda);
  } while (norm(update - b) > 1e-10);
  return b;
}

colvec Fit::l2grad(mat x, colvec y, colvec b) {
  colvec eta = x * b;
  colvec resid = y - eta;
  return (x.t() / x.n_rows) * (resid);
}
double Fit::l2loss(mat x, colvec y, colvec b) {
  double n = x.n_rows;
  colvec eta = x * b;
  colvec resid = y - eta;
  return (1 / n) * norm(resid, 2) * norm(resid, 2);
}
colvec Fit::loggrad(mat x, colvec y, colvec b) {
  double n = x.n_rows;
  colvec ones;
  ones = colvec(n, 1);
  ones.ones();

  colvec expeta = exp(x * b);

  return (-1 * x.t() / n) * (expeta / (ones + expeta) - y);
}

double Fit::logloss(mat x, colvec y, colvec b) {
  double n = x.n_rows;
  colvec ones;
  ones = colvec(n, 1);
  double loss = 0;
  colvec eta = x * b;
  loss = as_scalar(ones.t() * log(ones + exp(eta)) - y.t() * (eta));
  return loss;
}

/// @brief Backtracking line search step iteration.
///@param[in] x
/// Covariate matrix
///@param[in] y
/// Outcome Vector
///@param[in] b
/// Beta values from precios step.
///@param[in] lambda
/// Single lambda value to use
///@returns Fitted beta values for next step.
colvec Fit::fit(mat x, colvec y, colvec b, double lambda) {
  double t = 2.0;
  double prox;
  double actual;
  double loss;
  colvec bnew;
  colvec grad;
  if (loss_type_ == "log") {
    grad = loggrad(x, y, b);
    loss = logloss(x, y, b);
    do {
      t *= .5;
      bnew = Fit::soft_threshold(b + t * grad, t * lambda);
      prox = loss + as_scalar(grad.t() * (bnew - b)) +
             (1 / (2 * t)) * (norm(bnew - b, 2) * norm(bnew - b, 2));
      actual = logloss(x, y, bnew);
    } while (actual > prox);
  } else if (loss_type_ == "l2") {
    colvec grad = l2grad(x, y, b);
    double loss = l2loss(x, y, b);
    do {
      t *= .5;
      bnew = Fit::soft_threshold(b + t * grad, t * lambda);

      prox = loss + as_scalar(grad.t() * (bnew - b)) +
             (1 / (2 * t)) * (norm(bnew - b, 2) * norm(bnew - b, 2));
      actual = l2loss(x, y, bnew);
    } while (actual > prox);
  } else {
    throw std::runtime_error("Loss not supported.");
  }

  return bnew;
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
