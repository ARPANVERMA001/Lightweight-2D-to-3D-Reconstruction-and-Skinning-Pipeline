#include "flat_intersection_biquadratic.h"
#include <Eigen/SVD>
#include <Eigen/QR>
#include <iostream>
#include <cassert>

Eigen::VectorXd FlatIntersectionBiquadratic::pack(const Eigen::MatrixXd& W, int poses, int handles) {
    return Eigen::Map<const Eigen::VectorXd>(W.data(), W.size());
}

Eigen::MatrixXd FlatIntersectionBiquadratic::unpack(const Eigen::VectorXd& x, int poses, int handles) {
    assert(x.size() == 12 * poses * handles);
    return Eigen::Map<const Eigen::MatrixXd>(x.data(), 12 * poses, handles);
}

Eigen::MatrixXd FlatIntersectionBiquadratic::repeated_block_diag_times_matrix(
    const Eigen::MatrixXd& block, const Eigen::MatrixXd& matrix) 
{
    Eigen::Map<const Eigen::MatrixXd> reshaped_matrix(
        matrix.data(),
        block.cols(),
        matrix.rows() * matrix.cols() / block.cols()
    );
    
    Eigen::MatrixXd result = block * reshaped_matrix;
    
    return Eigen::Map<Eigen::MatrixXd>(
        result.data(),
        result.rows() * matrix.cols() / block.cols(),
        matrix.cols()
    );
}

FlatIntersectionBiquadratic::QuadraticExpression FlatIntersectionBiquadratic::quadratic_for_z(
    const Eigen::MatrixXd& W,
    const Eigen::VectorXd& v,
    const Eigen::VectorXd& vprime) 
{
    Eigen::MatrixXd v_mat = v.transpose();
    Eigen::MatrixXd VW = repeated_block_diag_times_matrix(v_mat, W);
    
    QuadraticExpression result;
    result.Q = VW.transpose() * VW;
    result.L = -2.0 * vprime.transpose() * VW;
    result.C = vprime.dot(vprime);
    
    return result;
}

FlatIntersectionBiquadratic::SolveForZResult FlatIntersectionBiquadratic::solve_for_z(
    const Eigen::MatrixXd& W,
    const Eigen::VectorXd& v,
    const Eigen::VectorXd& vprime,
    bool return_energy,
    bool use_pseudoinverse) 
{
    auto [Q, L, C] = quadratic_for_z(W, v, vprime);
    
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Q);
    double smallest_sv = svd.singularValues()(svd.singularValues().size()-1);
    
    int handles = L.size();
    Eigen::MatrixXd Qbig(Q.rows() + 1, Q.cols() + 1);
    Qbig.setZero();
    Qbig.topLeftCorner(Q.rows(), Q.cols()) = Q;
    Qbig.bottomLeftCorner(1, Q.cols()).setOnes();
    Qbig.topRightCorner(Q.rows(), 1).setOnes();
    
    Eigen::VectorXd rhs(L.size() + 1);
    rhs.head(L.size()) = -0.5 * L;
    rhs(L.size()) = 1.0;
    
    Eigen::VectorXd solution;
    if (use_pseudoinverse) {
        solution = Qbig.completeOrthogonalDecomposition().solve(rhs);
    } else {
        solution = Qbig.colPivHouseholderQr().solve(rhs);
    }
    
    SolveForZResult result;
    result.z = solution.head(handles);
    result.smallest_singular_value = smallest_sv;
    
    if (return_energy) {
        result.energy = result.z.transpose() * Q * result.z + L.dot(result.z) + C;
    }
    
    return result;
}

FlatIntersectionBiquadratic::QuadraticExpression FlatIntersectionBiquadratic::quadratic_for_v(
    const Eigen::MatrixXd& W,
    const Eigen::VectorXd& z,
    const Eigen::VectorXd& vprime,
    bool nullspace) 
{
    if (nullspace) {
        throw std::runtime_error("Nullspace calculation not implemented");
    }
    
    Eigen::Map<const Eigen::MatrixXd> Taffine(
        (W * z).data(),
        vprime.size() / 3,
        4
    );
    
    Eigen::MatrixXd T = Taffine.leftCols(3);
    Eigen::VectorXd t = Taffine.col(3);
    Eigen::VectorXd b = t - vprime;
    
    QuadraticExpression result;
    result.Q = T.transpose() * T;
    result.L = 2.0 * T.transpose() * b;
    result.C = b.dot(b);
    
    return result;
}

Eigen::VectorXd FlatIntersectionBiquadratic::solve_for_v(
    const Eigen::MatrixXd& W,
    const Eigen::VectorXd& z,
    const Eigen::VectorXd& vprime,
    bool nullspace,
    bool return_energy,
    bool use_pseudoinverse) 
{
    auto [Q, L, C] = quadratic_for_v(W, z, vprime, nullspace);
    
    Eigen::VectorXd v;
    if (use_pseudoinverse) {
        v = Q.completeOrthogonalDecomposition().solve(-0.5 * L);
    } else {
        v = Q.colPivHouseholderQr().solve(-0.5 * L);
    }
    
    if (return_energy) {
        double E = v.transpose() * Q * v + L.dot(v) + C;
        std::cout << "Energy after solve_for_v: " << E << std::endl;
    }
    
    Eigen::VectorXd result(v.size() + 1);
    result << v, 1.0;
    
    return result;
}

FlatIntersectionBiquadratic::LinearSystemForW FlatIntersectionBiquadratic::linear_matrix_equation_for_W(
    const Eigen::VectorXd& v,
    const Eigen::VectorXd& vprime,
    const Eigen::VectorXd& z) 
{
    Eigen::MatrixXd v_mat = v.transpose();
    
    LinearSystemForW result;
    result.A = v.transpose() * v;
    result.B = z * z.transpose();
    result.Y = repeated_block_diag_times_matrix(v_mat.transpose(), vprime) * z.transpose();
    
    return result;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> FlatIntersectionBiquadratic::zero_system_for_W(
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& B,
    const Eigen::MatrixXd& Y) 
{
    Eigen::MatrixXd system = Eigen::MatrixXd::Zero(B.cols() * A.rows(), B.rows() * A.cols());
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(Y.rows(), Y.cols());
    return {system, rhs};
}

void FlatIntersectionBiquadratic::accumulate_system_for_W(
    Eigen::MatrixXd& system,
    Eigen::MatrixXd& rhs,
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& B,
    const Eigen::MatrixXd& Y,
    double weight) 
{
    system += weight * Eigen::kroneckerProduct(A, B.transpose());
    rhs -= weight * Y;
}

Eigen::MatrixXd FlatIntersectionBiquadratic::solve_for_W(
    const std::vector<Eigen::MatrixXd>& As,
    const std::vector<Eigen::MatrixXd>& Bs,
    const std::vector<Eigen::MatrixXd>& Ys,
    bool use_pseudoinverse,
    const Eigen::MatrixXd* projection) 
{
    assert(As.size() == Bs.size() && As.size() == Ys.size() && !As.empty());
    assert(Ys[0].rows() % 12 == 0);
    
    int poses = Ys[0].rows() / 12;
    
    auto [system, rhs] = zero_system_for_W(As[0], Bs[0], Ys[0]);
    
    for (size_t i = 0; i < As.size(); i++) {
        accumulate_system_for_W(system, rhs, As[i], Bs[i], Ys[i], 1.0);
    }
    
    Eigen::MatrixXd W = solve_for_W_with_system(system, rhs, use_pseudoinverse);
    
    if (projection) {
        W = W * (*projection);
    }
    
    return W;
}

Eigen::MatrixXd FlatIntersectionBiquadratic::solve_for_W_with_system(
    const Eigen::MatrixXd& system,
    const Eigen::MatrixXd& rhs,
    bool use_pseudoinverse) 
{
    Eigen::MatrixXd W;
    if (use_pseudoinverse) {
        W = system.completeOrthogonalDecomposition().solve(rhs);
    } else {
        W = system.colPivHouseholderQr().solve(rhs);
    }
    
    Eigen::VectorXd col_means = W.colwise().mean();
    Eigen::MatrixXd W_centered = W.rowwise() - col_means.transpose();
    Eigen::VectorXd col_norms = W_centered.colwise().norm();
    std::cout << "W column norm: " << col_norms.transpose() << std::endl;
    
    return W;
}