#include <armadillo>
#include <iostream>
#include <map>
#include <string>
#include "optimize.hpp"

using namespace arma;
void Optimize::SetParams(colvec lambda,std::string loss_type, int num_params){
  lambda_=lambda;
  loss_type_=loss_type;
  colvec b(num_params,1);
  b.fill(0);
  b_=b;
}

colvec Optimize::partition_step(mat training_data, colvec training_outcomes,
                      mat testing_data, colvec testing_outcomes){
                      colvec results(lambda_.size(),1);
                      b_.fill(0);
                      for(int i=0;i<lambda_.size();i++)
                      {
                        results(i)=l2loss(training_data,training_outcomes,testing_data,testing_outcomes,b_,lambda_(i));
                      }
                      return results;
                      }

double Optimize::l2loss(mat training_data, colvec training_outcomes,
              mat testing_data, colvec testing_outcomes, colvec coeffs, double lambda){
        coeffs=Optimize::l2ggd(training_data,training_outcomes,coeffs, lambda);
        return sum(square(testing_data*coeffs - testing_outcomes))/testing_data.n_rows;;
}
double logloss(mat training_data, colvec training_outcomes,
              mat testing_data, colvec testing_outcomes, colvec coeffs, double lambda){

              };

colvec Optimize::l2ggd(mat x, colvec y, colvec b, double lambda){
    colvec update;
    update = b+(.1)*(x.t()/x.n_rows)*(y-(x * b));
    for (unsigned int i = 0; i < update.size(); i++) {
      update(i) = Optimize::soft_threshold(update(i), .1 * lambda);
    }
    if (norm(update - b) < 1e-3) {
      b = update;
      return b;
    } else {
      b = update;


      b = l2ggd(x,y,b,lambda);
      return b;
    }
}

double Optimize::soft_threshold(double x, double lambda) {
    if (x > lambda)
      return x - lambda;
    else if (x < -lambda)
      return x + lambda;
    else
      return 0;
  }
