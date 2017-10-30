#include <armadillo>
#include <iostream>
#include "optimize.cpp"
using namespace arma;
class Partition{
 private:
 int k_counter_;
 public:
  mat x_;
  colvec y_;
  colvec b_;
  colvec lambda_;
  mat results_;
  double t_;
  int k_;


  /// @brief Constructor for partition object with no prespecified lambda
  /// value.
  /// @param[in] x
  /// Data for predictor variables. (nxp)
  /// @param[in] y
  /// Column vector of response data. (p)
  /// @param[in] beta
  /// Initial values for coefficients. (p)
  /// @param[in] t
  /// Step size.
  /// @param[in] k
  /// Number of groups in partition.
  Partition(mat x, colvec y, colvec beta, double t, int k) : x_(x), y_(y), b_(beta), t_(t), k_(k) {
    double lmax = CalcLmax(x_, y_);
    lambda_ = linspace<vec>(lmax, lmax / 20, 100);
    k_counter_=0;
    results_=mat(lambda_.size(),k);
  }

  void IteratePartition(){
    mat x_k1=x_.rows(0,k_counter_*x_.n_rows/k_ );
    mat x_k2=x_.rows((k_counter_+1)*x_.n_rows/k_ -1,x_.n_rows-1);
    mat x_k=join_cols(x_k1,x_k2);
    colvec y_k1=y_.subvec(0,k_counter_*x_.n_rows/k_);
    colvec y_k2=y_.subvec((k_counter_+1)*x_.n_rows/k_ -1 ,x_.n_rows-1);
    colvec y_k=join_cols<mat>(y_k1,y_k2);
    colvec b_k(1000, 1);
    b_k.fill(0);
    for(int i=0;i<lambda_.size();i++){
    b_k=ggd(x_k,y_k,b_k,lambda_(i),t_);
    if(k_counter_==k_-1){
    x_k1=x_.rows(0,x_.n_rows/k_-1);
    y_k1=y_.subvec(0,x_.n_rows/k_-1);
    }
    else{
      x_k1=x_.rows(k_counter_*x_.n_rows/k_, (k_counter_+1)*x_.n_rows/k_ -1);
      y_k1=y_.subvec(k_counter_*x_.n_rows/k_, (k_counter_+1)*x_.n_rows/k_ -1);
    }
    results_(i,k_counter_)=pred_error(x_k1,b_k,y_k1);
    }
    k_counter_++;
  }
  colvec PartitionCycle(){
    while(k_counter_<k_){
      IteratePartition();
    }
    int maxindex=0;
    double temp;
    double best=1e20;
    for(int i=0; i< results_.n_rows;i++ ){
      temp= mean(results_.row(i));
      if(temp<best){
        best=temp;
        maxindex=i;
      }
    }
    cout<<"Best lambda value is: " << lambda_(maxindex)<< ". With average prediction error of "<< best<<"."<<endl;
    return ggd(x_,y_,b_,lambda_(maxindex),t_);
  }
    double soft_threshold(double x, double lambda) {
    if (x > lambda)
      return x - lambda;
    else if (x < -lambda)
      return x + lambda;
    else
      return 0;
  }
  double pred_error(mat x, colvec b, colvec y){
    return norm(x*b-y,2);
  }
  colvec ggd(mat x, colvec y, colvec b, double lambda, double t) {
    /// We want b= b+(t/N)x'(y-xb)
    colvec update;
    update = b+(t)*(x.t()/x.n_rows)*(y-(x * b));
    for (unsigned int i = 0; i < update.size(); i++) {
      update(i) = soft_threshold(update(i), t * lambda);
    }
    if (norm(update - b) < 1e-4) {
      b = update;
      return b;
    } else {
      b = update;
      b = ggd(x,y,b,lambda,t);
      return b;
    }
  }
double CalcLmax(mat x, colvec y){
  double lmax=0;
  double l;
  for(int i=0;i<x.n_cols;i++){
    l=dot(x.col(i),y)/x.n_rows;
    if(l>lmax) lmax=l;
  }
  return lmax;
}
};


