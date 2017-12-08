#include "cv.hpp"
#include <armadillo>
#include <iostream>
#include "fit.hpp"
using namespace arma;
/// @brief Constructor for partition object with no prespecified lambda
/// value.
/// @param[in] x
/// Data for predictor variables. (nxp)
/// @param[in] y
/// Column vector of response data. (p)
/// @param[in] k
/// Number of groups in partition.
CV::CV(mat x, colvec y, int k, std::string loss_type) : x_(x), y_(y) {
  double lmax = CalcLmax(x_, y_);
  lambda_ = linspace<vec>(lmax, lmax / 20, 100);
  opt.SetParams(loss_type);
  partition_counter_ = 0;
  partition_size_ = x_.n_rows / k;
  k_ = k;
  results_ = mat(100, k);
}
/// @brief Constructor for partition object with prespecified lambda
/// value.
/// @param[in] x
/// Data for predictor variables. (nxp)
/// @param[in] y
/// Column vector of response data. (p)
/// @param[in] k
/// Number of groups in partition.
/// @param[in] lambda
/// Prespecified lambda value.
/// @param[in] loss_type
/// Type of loss.
CV::CV(mat x, colvec y, int k, colvec lambda, std::string loss_type)
    : x_(x), y_(y) {
  double lmax = CalcLmax(x_, y_);
  colvec lambda_ = lambda;
  opt.SetParams(loss_type);
  partition_counter_ = 0;
  partition_size_ = x_.n_rows / k;
  results_ = mat(100, k);
}
///@brief Partitions data based on current partition step in CV provess.
void CV::IteratePartition() {
  // Partition x based on partition_counter_ values.
  mat test_x = x_.rows(partition_counter_ * partition_size_,
                       (partition_counter_ + 1) * (partition_size_)-1);
  mat train_x = x_;
  train_x.shed_rows(partition_counter_ * partition_size_,
                    (partition_counter_ + 1) * (partition_size_)-1);

  // Partition y based on partition_counter_ values.
  colvec test_y = y_.subvec(partition_counter_ * partition_size_,
                            (partition_counter_ + 1) * (partition_size_)-1);
  colvec train_y = y_;
  train_y.shed_rows(partition_counter_ * partition_size_,
                    (partition_counter_ + 1) * (partition_size_)-1);
  // Use fit object to create matrix of beta values for our lambda vector using
  // training data for this partition.
  mat betamat = opt.ggd(train_x, train_y, lambda_);
  // Get estimates for y values.
  betamat = betamat * test_x.t();
  // Fill results matrix with MSE for each lambda value.
  for (int i = 0; i < 100; i++) {
    results_(i, partition_counter_) =
        sum(square(betamat.row(i).t() - test_y)) / test_x.n_rows;
  }

  // cout<<"Partition "<<partition_counter_<<" complete."<<endl;
  partition_counter_++;
}
/// @brief Wrapper for iterating IteratePartition() function.
void CV::PartitionCycle() {
  while (partition_counter_ < k_) {
    IteratePartition();
    cout << "Partition " << partition_counter_ << " complete." << endl;
  }
}

/// @brief Finds the minimum average MSE across all lambda values.
/// @returns double of best lambda value.
double CV::ReturnBestLambda() {
  int maxindex = 0;
  double temp;
  double best = 1e20;
  for (int i = 0; i < results_.n_rows; i++) {
    temp = mean(results_.row(i));
    if (temp < best) {
      best = temp;
      maxindex = i;
    }
  }
  return lambda_(maxindex);
}

///@brief Calculates maximum lambda value to use for data.
///@param[in] x
/// Covariate data matrix
///@param[in] y
/// Outcome data vector
///@return Max lambda value
double CV::CalcLmax(mat x, colvec y) {
  double lmax = 0;
  double l;
  for (int i = 0; i < x.n_cols; i++) {
    l = dot(x.col(i), y) / x.n_rows;
    if (l > lmax) lmax = l;
  }
  return lmax;
}
