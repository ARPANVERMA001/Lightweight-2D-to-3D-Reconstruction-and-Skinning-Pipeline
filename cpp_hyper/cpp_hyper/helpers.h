// helpers.h

#ifndef HELPERS_H
#define HELPERS_H

#include <Eigen/Dense>
#include <tuple>
#include <string>

// Global variable to store the previous projection
extern Eigen::MatrixXd prev_Fw_projected;

// Function to perform PCA
Eigen::MatrixXd performPCA(const Eigen::MatrixXd &data, int n_components);

// Function to perform Procrustes alignment
std::tuple<Eigen::MatrixXd, double> procrustesAlignment(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B);

// Function to write scatter plot data to CSV
void writeScatterCSV(const std::string &filename, const Eigen::VectorXd &x, const Eigen::VectorXd &y, const Eigen::VectorXd &z);

#endif // HELPERS_H