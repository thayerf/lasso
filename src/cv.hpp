#ifndef PARTITION
#define PARTITION

#include <armadillo>
#include <iostream>
#include "fit.hpp"
using namespace arma;
class CV {
 private:
  int partition_counter_;
  int partition_size_;

 public:
  mat x_;
  int k_;
  colvec y_;
  colvec b_;
  Fit opt;
  colvec lambda_;
  mat results_;
  CV(mat x, colvec y, int k, colvec lambda, std::string loss_type);
  CV(mat x, colvec y, int k, std::string loss_type);
  void IteratePartition();
  void PartitionCycle();
  double ReturnBestLambda();
  double CalcLmax(mat x, colvec y);
};

#endif  // PARTITION
