#pragma once

#include <Eigen/Dense>
#include <vector>

class FlatIntersectionBiquadratic {
public:
    // Core data structures
    struct QuadraticExpression {
        Eigen::MatrixXd Q;
        Eigen::VectorXd L;
        double C;
    };

    struct SolveForZResult {
        Eigen::VectorXd z;
        double smallest_singular_value;
        double energy;
    };

    struct LinearSystemForW {
        Eigen::MatrixXd A;
        Eigen::MatrixXd B;
        Eigen::MatrixXd Y;
    };

    // Pack/Unpack functions
    static Eigen::VectorXd pack(const Eigen::MatrixXd& W, int poses, int handles);
    static Eigen::MatrixXd unpack(const Eigen::VectorXd& x, int poses, int handles);

    // Matrix manipulation functions
    static Eigen::MatrixXd repeated_block_diag_times_matrix(
        const Eigen::MatrixXd& block, 
        const Eigen::MatrixXd& matrix
    );

    // Core computation functions
    static QuadraticExpression quadratic_for_z(
        const Eigen::MatrixXd& W,
        const Eigen::VectorXd& v,
        const Eigen::VectorXd& vprime
    );

    static SolveForZResult solve_for_z(
        const Eigen::MatrixXd& W,
        const Eigen::VectorXd& v,
        const Eigen::VectorXd& vprime,
        bool return_energy = false,
        bool use_pseudoinverse = true
    );

    static QuadraticExpression quadratic_for_v(
        const Eigen::MatrixXd& W,
        const Eigen::VectorXd& z,
        const Eigen::VectorXd& vprime,
        bool nullspace = false
    );

    static Eigen::VectorXd solve_for_v(
        const Eigen::MatrixXd& W,
        const Eigen::VectorXd& z,
        const Eigen::VectorXd& vprime,
        bool nullspace = false,
        bool return_energy = false,
        bool use_pseudoinverse = false
    );

    static LinearSystemForW linear_matrix_equation_for_W(
        const Eigen::VectorXd& v,
        const Eigen::VectorXd& vprime,
        const Eigen::VectorXd& z
    );

    static std::pair<Eigen::MatrixXd, Eigen::MatrixXd> zero_system_for_W(
        const Eigen::MatrixXd& A,
        const Eigen::MatrixXd& B,
        const Eigen::MatrixXd& Y
    );

    static void accumulate_system_for_W(
        Eigen::MatrixXd& system,
        Eigen::MatrixXd& rhs,
        const Eigen::MatrixXd& A,
        const Eigen::MatrixXd& B,
        const Eigen::MatrixXd& Y,
        double weight
    );

    static Eigen::MatrixXd solve_for_W(
        const std::vector<Eigen::MatrixXd>& As,
        const std::vector<Eigen::MatrixXd>& Bs,
        const std::vector<Eigen::MatrixXd>& Ys,
        bool use_pseudoinverse = true,
        const Eigen::MatrixXd* projection = nullptr
    );

    static Eigen::MatrixXd solve_for_W_with_system(
        const Eigen::MatrixXd& system,
        const Eigen::MatrixXd& rhs,
        bool use_pseudoinverse = true
    );
};