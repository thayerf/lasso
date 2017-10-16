#include <iostream>
#include <armadillo>
using namespace arma;
class GGD{
public:
mat x_;
colvec y_;
colvec b_;
colvec lambda_;
double t_;

/// @brief Constructor for gradient decent object.
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
GGD(mat x, colvec y, colvec beta, double t) : x_(x),y_(y),b_(beta),t_(t)
{
  double lmax= CalcLmax(x_,y_);
  lambda_=linspace<vec>(lmax/2,lmax,100);
}

GGD(mat x, colvec y, colvec beta,colvec lambda, double t) : x_(x),y_(y),b_(beta), lambda_(lambda),t_(t)
{
}

double CalcLmax(mat x, colvec y){
 double lmax=0;
 double l;
 for(int i=0;i<x.n_cols;i++){
  l=dot(x.col(i),y)/x.n_rows;
  if(l>lmax) lmax=l;
 }
 return lmax-.01;
}

GGD lasso(GGD k){
///TODO: Fix this. Diverges.
  colvec temp;
  temp=k.x_*k.b_;
  temp=k.y_-temp;
  temp= k.x_.t()*temp;
  temp*=k.t_;
  temp+=k.b_;
  std::cout<<norm(temp,1)<<std::endl;
  for(unsigned int i=0;i<temp.size();i++){
    temp(i)=soft_threshold(temp(i),k.t_*k.lambda_(0));
  }
  if(std::abs(norm(temp,1)-norm(k.b_,1)) <.02) {
  k.b_=temp;
  return k;}
  else{
  k.b_=temp;
  k=lasso(k);
  }
}
double pred_error(GGD k){
  colvec t= k.x_*k.b_;
  t-=k.y_;
  return norm(t,1);
}

double soft_threshold(double x, double lambda){
  if(x > lambda) return x-lambda;
  else if(x < -lambda) return x+lambda;
  else return 0;
}
};
