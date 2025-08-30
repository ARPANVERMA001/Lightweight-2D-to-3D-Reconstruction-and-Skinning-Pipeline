#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <Eigen/Dense>
#include "flat_metrics.h"
// Helper functions to mimic Python's array logic:

// clip each element of vector to [-1,1]
static void clip_in_place(Eigen::VectorXd &v, double low, double high) {
    for (int i=0; i<v.size(); i++) {
        if(v[i]<low) v[i]=low;
        if(v[i]>high) v[i]=high;
    }
}

// searchsorted on reversed array for threshold
// Python code: stop_s = len(s) - np.searchsorted(s[::-1], threshold)
// s is singular values sorted descending. s[::-1] ascending.
// We do the same by constructing a reversed vector and using lower_bound.
static int search_stop_s(const Eigen::VectorXd &s, double threshold) {
    std::vector<double> s_rev(s.size());
    for (int i=0; i<s.size(); i++) s_rev[i]=s[s.size()-1-i];
    auto it = std::lower_bound(s_rev.begin(), s_rev.end(), threshold);
    int idx=(int)(it - s_rev.begin());
    int stop_s=(int)s.size()-idx;
    return stop_s;
}

// orthonormalize(A)
Eigen::MatrixXd orthonormalize(const Eigen::MatrixXd &A, double threshold) {
    // A: n x m
    // SVD of A^T: A^T = U s V, A = V^T s U^T
    // Actually code: U,s,V = svd(A^T)
    // A^T shape: (m,n)
    // full_matrices=False means U,V minimal shapes
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A.transpose(), Eigen::ComputeThinU|Eigen::ComputeThinV);
    Eigen::VectorXd s = svd.singularValues();
    int stop_s = search_stop_s(s, threshold);
    // V from A^T's SVD: A^T = U s V
    // We want V[:stop_s].T
    // Here V = svd.matrixV() => shape (n,n)
    // We take first stop_s rows of V: V.topRows(stop_s), shape (stop_s,n)
    // Then transpose it to get (n,stop_s)
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::MatrixXd R = V.topRows(stop_s).transpose(); 
    // R is (n,stop_s)
    return R;
}

// principal_cosangles(A,B,orthonormal=False)
Eigen::VectorXd principal_cosangles(Eigen::MatrixXd A, Eigen::MatrixXd B, bool orthonormal) {
    if(!orthonormal) {
        A = orthonormalize(A);
        B = orthonormalize(B);
    }
    // SVD of A^T B
    // principal angles = singular values of A^T B
    Eigen::MatrixXd M = A.transpose()*B;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU|Eigen::ComputeThinV);
    Eigen::VectorXd principal_angles = svd.singularValues(); 
    // Append zeros if dimension mismatches:
    int diff = std::max(A.cols(), B.cols()) - std::min(A.cols(), B.cols());
    if (diff>0) {
        Eigen::VectorXd extended(principal_angles.size()+diff);
        extended << principal_angles, Eigen::VectorXd::Zero(diff);
        return extended;
    } else {
        return principal_angles;
    }
}

// principal_angles(A,B, orthonormal=False)
Eigen::VectorXd principal_angles_fn(Eigen::MatrixXd A, Eigen::MatrixXd B, bool orthonormal) {
    Eigen::VectorXd cosangles = principal_cosangles(A,B,orthonormal);
    // clip to [-1,1]
    clip_in_place(cosangles,-1.0,1.0);
    Eigen::VectorXd angles(cosangles.size());
    for (int i=0; i<cosangles.size(); i++) {
        angles[i]=std::acos(cosangles[i]);
    }
    return angles;
}

// canonical_point(p,B)
Eigen::VectorXd canonical_point(const Eigen::VectorXd &p, const Eigen::MatrixXd &B) {
    // p_closest = p - B (B'B)^{-1} B' p
    // Solve (B'B) z = B'p  => z = (B'B)^{-1} B'p
    // p_closest = p - B z
    Eigen::MatrixXd BtB = B.transpose()*B;
    Eigen::VectorXd Btp = B.transpose()*p;
    Eigen::VectorXd z = BtB.fullPivLu().solve(Btp);
    Eigen::VectorXd p_closest = p - B*z;
    return p_closest;
}

// distance_between_flats(p1,B1,p2,B2)
double distance_between_flats(Eigen::VectorXd p1, Eigen::MatrixXd B1, Eigen::VectorXd p2, Eigen::MatrixXd B2) {
    B1 = orthonormalize(B1);
    B2 = orthonormalize(B2);

    auto parallel_projector = [&](const Eigen::MatrixXd &B)->Eigen::MatrixXd {
        int n=(int)B.rows();
        Eigen::MatrixXd P_Bortho = -B*(B.transpose());
        for (int i=0; i<n; i++) {
            P_Bortho(i,i)+=1;
        }
        return P_Bortho;
    };

    Eigen::MatrixXd P_B1ortho = parallel_projector(B1);
    Eigen::MatrixXd P_B2ortho = parallel_projector(B2);
    // orthogonal_to_B1_and_B2 = 2 P_B1ortho (P_B1ortho + P_B2ortho)^+ P_B2ortho
    // Use pseudo-inverse with SVD or just fullPivLu().solve:
    // pinv(P_B1ortho+P_B2ortho)
    Eigen::MatrixXd S = P_B1ortho + P_B2ortho;
    Eigen::MatrixXd Sinv = S.completeOrthogonalDecomposition().pseudoInverse();

    Eigen::MatrixXd orthogonal_to_B = (2.0 * P_B1ortho) * (Sinv * P_B2ortho);
    Eigen::VectorXd d = orthogonal_to_B*(p1 - p2);
    return d.norm();
}

// optimal_p_given_B_for_flats_ortho(B, flats)
// flats: vector of pairs (A,a)
// A: matrix, a: vector
// B not necessarily orthonormal
// We implement formula from code comments
Eigen::VectorXd optimal_p_given_B_for_flats_ortho(const Eigen::MatrixXd &B,
                                                         const std::vector<std::pair<Eigen::MatrixXd,Eigen::VectorXd>> &flats) {
    int n = (int)flats[0].first.cols(); // dimension from A
    Eigen::MatrixXd system = Eigen::MatrixXd::Zero(n,n);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);

    // Q = I - A B (B'A'A B)^-1 B' A'
    // But code uses a trick:
    // Q computed as:
    // Q = ...
    // Wait carefully: The code tries:
    // Q is constructed with lstsq:
    // Q = I - A B (B'A' A B)^{-1} B' A'
    // In code they do:
    // AB = A B
    // Q = I - A B ((B' A' A B)^{-1}) B' A'
    // They do: linalg.lstsq to solve (B' A' A B)
    // We'll do fullPivLu().solve

    // Actually code snippet inside loop:
    // AB = A B
    // Q = I - A B inv(B' A' A B) B' A'
    // They form Q by:
    // Q = -A B [inv(B'A'A B) B'A'] ; Q diag += 1
    // Actually code:
    // Q = -dot(AB, linalg.lstsq(dot(AB.T,AB),AB.T )[0])
    // Q[diag_indices_from(Q)] += 1

    // Wait carefully:
    // M = Ap - Aa - AB inv(...) B'(Ap - Aa)
    // They do a sum over flats of A'Q A, etc.

    for (auto &fa: flats) {
        const Eigen::MatrixXd &A = fa.first;
        const Eigen::VectorXd &a = fa.second;

        Eigen::MatrixXd AB = A*B; // (r x n)*(n x k) = (r x k)
        // Solve (AB^T AB) Z = AB^T: This gives pseudo-inverse part:
        // Actually they do: Z = lstsq(AB^T AB, AB^T)
        // Let's do:
        Eigen::MatrixXd ABtAB = AB.transpose()*AB;
        Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(ABtAB);
        Eigen::MatrixXd Z = cod.pseudoInverse()*AB.transpose(); // This is (B' A' A B)^{-1} B' A'

        // Q = I - A B Z A'
        // Let's form Q:
        int rr = (int)A.rows();
        int cc = (int)A.cols();
        // Q must be (r x r), same as A*A'
        // Actually Q same shape as (A?), from code Q is same shape as A*A', i.e. (r x r)
        Eigen::MatrixXd Q = -AB * Z; // (r,k)*(k,r)=(r,r)
        for (int i=0; i<rr; i++) Q(i,i)+=1.0; // Q diag +=1

        // AQA = A' Q A (n x n)
        Eigen::MatrixXd AQA = A.transpose()*Q*A;
        system += AQA;
        rhs += AQA*a;
    }

    // solve system p
    // use lstsq to get minimal norm solution
    Eigen::VectorXd p = system.completeOrthogonalDecomposition().solve(rhs);
    return p;
}

// The code includes tests in Python. We won't replicate the test harness here.
// If needed, write your own tests in C++.