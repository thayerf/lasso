#include <armadillo>
#include <iostream>
using namespace arma;
class GGD {
 public:
  mat x_;
  colvec y_;
  colvec b_;
  colvec lambda_;
  double t_;

  /// @brief Constructor for gradient decent object with no prespecified lambda
  /// value.
  /// @param[in] x
  /// Data for predictor variables. (nxp)
  /// @param[in] y
  /// Column vector of response data. (p)
  /// @param[in] beta
  /// Initial values for coefficients. (p)
  /// @param[in] t
  /// Step size.
  GGD(mat x, colvec y, colvec beta, double t) : x_(x), y_(y), b_(beta), t_(t) {
    double lmax = CalcLmax(x_, y_);
    lambda_ = linspace<vec>(lmax, lmax / 2, 30);
  }

  /// @brief Constructor for gradient decent object with prespecified lambda
  /// value.
  /// @param[in] x
  /// Data for predictor variables. (nxp)
  /// @param[in] y
  /// Column vector of response data. (p)
  /// @param[in] beta
  /// Initial values for coefficients. (p)
  /// @param[in] lambda
  /// Values for lambda.
  /// @param[in] t
  /// Step size.
  GGD(mat x, colvec y, colvec beta, colvec lambda, double t)
      : x_(x), y_(y), b_(beta), lambda_(lambda), t_(t) {}

  double CalcLmax(mat x, colvec y) {
    double lmax = 0;
    double l;
    for (int i = 0; i < x.n_cols; i++) {
      l = dot(x.col(i), y) / x.n_rows;
      if (l > lmax) lmax = l;
    }
    return lmax - .01;
  }

  /// @brief Recursive function for lasso using GGD.
  /// @param[in] k
  /// GGD object on which to perform lasso.
  GGD lasso(GGD k) {
    /// @TODO: Fix this. Diverges.

    /// We want b= b+(t/N)x'(y-xb)
    colvec update;
    update = k.b_+(k.t_)*k.x_.t()*(k.y_-(1/k.b_.size())*(k.x_ * k.b_));

    for (unsigned int i = 0; i < update.size(); i++) {
      update(i) = soft_threshold(update(i), k.t_ * k.lambda_(0));
    }
    cout<<update[0]<<","<<update[1]<<","<<update[2]<<endl;
    cout<<std::abs(norm(update, 1) - norm(k.b_, 1))<<endl;
    if (std::abs(norm(update, 1) - norm(k.b_, 1)) < .2) {
      k.b_ = update;
      return k;
    } else {
      k.b_ = update;
      k = lasso(k);
      return k;
    }
  }
  /// @brief Calculates prediction error on itself (for now)
  /// @param[in] k
  /// GGD object on which to calculate prediction error.
  double pred_error(GGD k) {
    colvec t = k.x_ * k.b_;
    t -= k.y_;
    return norm(t, 1);
  }

  /// @brief Soft threshold function.
  double soft_threshold(double x, double lambda) {
    if (x > lambda)
      return x - lambda;
    else if (x < -lambda)
      return x + lambda;
    else
      return 0;
  }
};
