#ifndef SPACEMAPPER_H
#define SPACEMAPPER_H

#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>

class SpaceMapper {
public:
    Eigen::RowVectorXd Xavg_;
    Eigen::MatrixXd U_;
    Eigen::VectorXd s_;
    Eigen::MatrixXd V_;
    int stop_s;
    Eigen::VectorXd scale;

    SpaceMapper();

    Eigen::MatrixXd project(const Eigen::MatrixXd &correllated_poses) const;
    Eigen::MatrixXd unproject(const Eigen::MatrixXd &uncorrellated_poses) const;

    static int PCA_Dimension(const Eigen::MatrixXd &X, double threshold = 1e-6);
    static SpaceMapper Uncorrellated_Space(const Eigen::MatrixXd &X, bool enable_scale = true, double threshold = 1e-6, int dimension = -1);
};

#endif // SPACEMAPPER_H