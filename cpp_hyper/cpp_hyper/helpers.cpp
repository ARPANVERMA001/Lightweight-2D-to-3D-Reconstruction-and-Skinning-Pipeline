// helpers.cpp

#include "helpers.h"
#include <iostream>
#include <fstream>
#include <stdexcept>

// Initialize the global variable
Eigen::MatrixXd prev_Fw_projected;

// Function to perform PCA using Eigen's SVD
Eigen::MatrixXd performPCA(const Eigen::MatrixXd &data, int n_components) {
    if(n_components <= 0 || n_components > data.cols()){
        throw std::invalid_argument("Invalid number of PCA components.");
    }

    // Center the data
    Eigen::MatrixXd centered = data.rowwise() - data.colwise().mean();
    
    // Compute SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinV);
    Eigen::MatrixXd V = svd.matrixV();
    
    // Select the top n_components
    Eigen::MatrixXd principalComponents = V.leftCols(n_components);
    
    // Project the data
    Eigen::MatrixXd projected = centered * principalComponents;
    
    return projected;
}

// Function to perform Procrustes alignment
std::tuple<Eigen::MatrixXd, double> procrustesAlignment(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B){
    if(A.rows() != B.rows() || A.cols() != B.cols()){
        throw std::invalid_argument("Procrustes alignment requires A and B to have the same dimensions.");
    }

    int m = A.rows();
    int n = A.cols();

    // Center the matrices
    Eigen::MatrixXd A_centered = A.rowwise() - A.colwise().mean();
    Eigen::MatrixXd B_centered = B.rowwise() - B.colwise().mean();

    // Compute the scaling
    double norm_A = A_centered.norm();
    double norm_B = B_centered.norm();
    if(norm_A == 0 || norm_B == 0){
        throw std::invalid_argument("Procrustes alignment requires non-zero norm.");
    }
    A_centered /= norm_A;
    B_centered /= norm_B;

    // Compute the optimal rotation using SVD
    Eigen::MatrixXd M = A_centered.adjoint() * B_centered;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();

    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(n, n);
    if( (svd.singularValues().array() < 0).any()){
        S(n-1, n-1) = -1;
    }

    Eigen::MatrixXd R = V * S * U.adjoint();

    // Compute disparity
    Eigen::MatrixXd B_aligned = (A_centered * R) * norm_B + B.colwise().mean();
    double disparity = (A - B_aligned).squaredNorm();

    return std::make_tuple(B_aligned, disparity);
}

// Function to write scatter plot data to CSV
void writeScatterCSV(const std::string &filename, const Eigen::VectorXd &x, const Eigen::VectorXd &y, const Eigen::VectorXd &z){
    std::ofstream file(filename);
    if(!file.is_open()){
        throw std::runtime_error("Cannot open file for writing scatter plot data: " + filename);
    }
    file << "x,y,z\n";
    for(int i=0; i<x.size(); ++i){
        file << x(i) << "," << y(i) << "," << z(i) << "\n";
    }
    file.close();
}