#include <armadillo>
#include <iostream>
#include <map>
#include <string>
#include "fit.hpp"

using namespace arma;
void Fit::SetParams(std::string loss_type){
  loss_type_=loss_type;
}

///@brief Proximal gradient decent for vector of lambda values.
///@param[in] x
/// Covariate matrix
///@param[in] y
/// Outcome Vector
///@param[in] lambda
/// Vector of lambda values to use
///@return Matrix containing beta values for all lambda values.
mat Fit::l2ggd(mat x, colvec y, colvec lambda){
    mat results;
    colvec b;
    b=colvec(x.n_cols,1);
    b.fill(0);
    results=mat(lambda.size(),x.n_cols);
    colvec update=b;

    for(int j=0;j<lambda.size();j++){
   do{
    b=update;
    update = b+(.1)*(x.t()/x.n_rows)*(y-(x * b));
    for (unsigned int i = 0; i < update.size(); i++) {
      update(i) = Fit::soft_threshold(update(i), .1 * lambda(j));
    }
   }
  while(norm(update - b) > 1e-5);
  results.row(j)=b.t();
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
colvec Fit::l2ggd(mat x, colvec y, double lambda){
    colvec b;
    b=colvec(x.n_cols,1);
    b.fill(0);
    colvec update=b;
    do{
    b=update;
    update = b+(.1)*(x.t()/x.n_rows)*(y-(x * b));
    for (unsigned int i = 0; i < update.size(); i++) {
      update(i) = Fit::soft_threshold(update(i), .1 * lambda);
    }
    }
    while(norm(update - b) > 1e-5);
      return b;
    }
///@brief Soft threshold function
double Fit::soft_threshold(double x, double lambda) {
    if (x > lambda)
      return x - lambda;
    else if (x < -lambda)
      return x + lambda;
    else
      return 0;
  }
