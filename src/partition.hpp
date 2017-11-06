#ifndef PARTITION
#define PARTITION

#include <armadillo>
#include <iostream>
#include "optimize.hpp"
using namespace arma;
class Partition{
 private:
 int partition_counter_;
 int partition_size_;
 public:
  mat x_;
  colvec y_;
  colvec b_;
  Optimize opt;
  mat results_;
  Partition(mat x, colvec y, int k);
  void IteratePartition();
  colvec PartitionCycle();
  double CalcLmax(mat x, colvec y);
};

#endif //PARTITION
