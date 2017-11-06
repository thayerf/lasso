#include <armadillo>
#include <iostream>
#include "optimize.hpp"
#include "partition.hpp"
using namespace arma;
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
  Partition::Partition(mat x, colvec y, int k) : x_(x), y_(y){
    double lmax = CalcLmax(x_, y_);
    colvec lambda= linspace<vec>(lmax, lmax / 20, 100);
    opt.SetParams(lambda,"l2",x_.n_cols);
    partition_counter_=0;
    partition_size_= x_.n_rows/k;
    results_=mat(100,k);
  }

  void Partition::IteratePartition(){
    mat test_x= x_.rows(partition_counter_*partition_size_,(partition_counter_+1)*(partition_size_)-1);
    mat train_x= x_;
    train_x.shed_rows(partition_counter_*partition_size_,(partition_counter_+1)*(partition_size_)-1);

    colvec test_y=y_.subvec(partition_counter_*partition_size_,(partition_counter_+1)*(partition_size_)-1);
    colvec train_y= y_;
    train_y.shed_rows(partition_counter_*partition_size_,(partition_counter_+1)*(partition_size_)-1);

    results_.col(partition_counter_)=opt.partition_step(train_x,train_y,test_x,test_y);
    cout<<"Partition "<<partition_counter_<<" complete."<<endl;
    partition_counter_++;
  }
  colvec Partition::PartitionCycle(){
    while(partition_counter_<x_.n_rows/partition_size_){
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
    cout<<"Best lambda value is: " << opt.lambda_(maxindex)<< ". With MSE of "<< best<<"."<<endl;
    cout<<results_<<endl;
    return opt.l2ggd(x_,y_,opt.b_,opt.lambda_(maxindex));
  }
double Partition::CalcLmax(mat x, colvec y){
  double lmax=0;
  double l;
  for(int i=0;i<x.n_cols;i++){
    l=dot(x.col(i),y)/x.n_rows;
    if(l>lmax) lmax=l;
  }
  return lmax;
}



