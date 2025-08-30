#ifndef FLAT_METRICS_H
#define FLAT_METRICS_H

#include <Eigen/Dense>
#include <vector>

// Orthonormalize a matrix A.
// Returns an orthonormal basis for the column space of A.
// Any singular values below 'threshold' are discarded.
Eigen::MatrixXd orthonormalize(const Eigen::MatrixXd &A, double threshold=1e-9);

// Compute the principal cosines of the angles between subspaces spanned by A and B.
// If 'orthonormal' is false, A and B will be orthonormalized before computation.
Eigen::VectorXd principal_cosangles(Eigen::MatrixXd A, Eigen::MatrixXd B, bool orthonormal=false);

// Compute the principal angles between subspaces spanned by A and B.
// Angles are in radians and returned in ascending order.
// If 'orthonormal' is false, A and B will be orthonormalized before computation.
Eigen::VectorXd principal_angles_fn(Eigen::MatrixXd A, Eigen::MatrixXd B, bool orthonormal=false);

// Compute the canonical point of 'p' with respect to the subspace spanned by columns of B.
// This is the point in the subspace closest to 'p'.
Eigen::VectorXd canonical_point(const Eigen::VectorXd &p, const Eigen::MatrixXd &B);

// Compute the distance between two flats (affine subspaces) defined by (p1, B1) and (p2, B2).
// 'p1' and 'p2' are points on the flats, and 'B1' and 'B2' are basis matrices.
// Returns the shortest distance between the two flats.
double distance_between_flats(Eigen::VectorXd p1, Eigen::MatrixXd B1, Eigen::VectorXd p2, Eigen::MatrixXd B2);

// Compute the optimal point 'p' given a basis 'B' and a set of flats.
// Each flat is represented as a pair of matrix 'A' and vector 'a'.
// This function solves for 'p' that minimizes the sum of squared distances to the flats.
Eigen::VectorXd optimal_p_given_B_for_flats_ortho(
    const Eigen::MatrixXd &B,
    const std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> &flats
);

#endif // FLAT_METRICS_H