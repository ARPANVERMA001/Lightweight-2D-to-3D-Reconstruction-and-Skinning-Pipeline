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
    Eigen::VectorXd scale; // can be empty if no scale

    SpaceMapper() : stop_s(0) {}

    Eigen::MatrixXd project(const Eigen::MatrixXd &correllated_poses) const {
        // correllated_poses: shape (N, features)
        // V_: shape (features, features)
        // We use V_.topRows(stop_s) is analogous to V[:stop_s], but in Eigen we must
        // think carefully: In python V is (samples x features), after SVD on Xp:
        // Python's V from np.linalg.svd is shape (features, features)
        // V[:stop_s] in python means take first stop_s rows of V if we treat V as (features, features).
        // Actually, V in SVD (U s V^T) from numpy: Xp = U s V.
        // np.linalg.svd returns V as (features, features). 
        // V[:stop_s].T would be (stop_s, features) transposed is (features, stop_s).
        // We'll do: V_.topRows(stop_s) gives us a (stop_s, features) matrix.
        // In python: (correllated_poses - Xavg) * V[:stop_s].T
        // In C++: (correllated_poses - Xavg_) * V_.topRows(stop_s).transpose()

        Eigen::MatrixXd centered = correllated_poses.rowwise() - Xavg_;
        Eigen::MatrixXd proj = centered * V_.topRows(stop_s).transpose(); // shape: (N, stop_s)

        if (scale.size() > 0) {
            // scale is (stop_s), broadcast multiply
            for (int i=0; i<stop_s; i++) {
                proj.col(i) = proj.col(i).array() * scale(i);
            }
        }
        return proj;
    }

    Eigen::MatrixXd unproject(const Eigen::MatrixXd &uncorrellated_poses) const {
        // uncorrellated_poses: shape (N, stop_s)
        // We do the inverse operation:
        // If scale is not None:
        //   (uncorrellated_poses / scale) * V[:stop_s] + Xavg
        // else:
        //   uncorrellated_poses * V[:stop_s] + Xavg

        Eigen::MatrixXd input = uncorrellated_poses;
        if (scale.size() > 0) {
            // divide each column by scale(i)
            for (int i=0; i<stop_s; i++) {
                input.col(i) = input.col(i).array() / scale(i);
            }
        }

        Eigen::MatrixXd recon = input * V_.topRows(stop_s); // shape (N, features)
        recon.rowwise() += Xavg_;
        return recon;
    }

    static int PCA_Dimension(const Eigen::MatrixXd &X, double threshold = 1e-6) {
        // Subtract average
        Eigen::RowVectorXd Xavg = X.colwise().mean();
        Eigen::MatrixXd Xp = X.rowwise() - Xavg;

        // SVD
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(Xp, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd s = svd.singularValues();

        // We need to find stop_s:
        // stop_s = len(s) - searchsorted(s[::-1], threshold)
        // s[::-1] is s in reverse order.
        // searchsorted with ascending order means find the position where threshold would fit.
        // We'll do: find how many are < threshold from the end.
        // s is sorted decreasing by SVD from largest to smallest? Actually, np.linalg.svd returns singular values sorted descending.
        // If s is descending, s[::-1] is ascending.
        // searchsorted(s[::-1], threshold) finds the index in ascending order where threshold fits.
        // If threshold is 1e-6, we find from the smallest end how many are less than threshold.
        // To replicate: Let's create a reversed copy and then use std::lower_bound.
        
        std::vector<double> s_rev(s.size());
        for (int i=0; i<s.size(); i++) s_rev[i] = s(s.size()-1 - i);
        
        // s_rev is ascending. We find the insertion point for threshold.
        auto it = std::lower_bound(s_rev.begin(), s_rev.end(), threshold);
        int idx = (int)(it - s_rev.begin());
        // idx is the position in ascending order. 
        // stop_s = len(s) - idx
        int stop_s = (int)s.size() - idx;
        return stop_s;
    }

    static SpaceMapper Uncorrellated_Space(const Eigen::MatrixXd &X, bool enable_scale=true, double threshold=1e-6, int dimension=-1) {
        // Check parameters
        if (threshold != 1e-6 && dimension != -1 && threshold != 0.0) {
            // The original code states:
            // if threshold is not None and dimension is not None:
            // please only set one
            if (dimension != -1) {
                throw std::runtime_error("Please only set one of threshold or dimension");
            }
        }

        SpaceMapper space_mapper;

        // Compute average
        Eigen::RowVectorXd Xavg = X.colwise().mean();
        Eigen::MatrixXd Xp = X.rowwise() - Xavg;
        space_mapper.Xavg_ = Xavg;

        // SVD
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(Xp, Eigen::ComputeThinU | Eigen::ComputeThinV);
        space_mapper.U_ = svd.matrixU();
        space_mapper.s_ = svd.singularValues();
        space_mapper.V_ = svd.matrixV().transpose(); 
        // Note: np.linalg.svd returns U, s, V with X = U * s * V.
        // In Eigen, JacobiSVD returns U and V so that X = U * s.asDiagonal() * V.transpose().
        // np: V is of shape (features, features), 
        // Eigen: V is same but we must be careful. matrixV() gives V, so Xp = U * s * V^T.
        // So to match Python: V in python is what we have as V_. In python code, final V_ is the V from SVD?
        // Actually python: U, s, V = np.linalg.svd(Xp), then Xp = U * diag(s) * V.
        // So Python's V is the same as Eigen's V (not transposed).
        // Wait carefully: np.linalg.svd returns V already as V (not V^T).
        // Eigen returns V so that X = U * S * V^T.
        // Python: Xp = U * S * V
        // So Python's V = Eigen's V^T. We must store V_ as python would:
        // python: V_ = V from svd is (features, features)
        // Eigen: matrixV() is (features, features) but it's the V in X=U*S*V^T. So V_eigen = V_python^T.
        // To get python's V we must take eigen's V and transpose it: 
        // Let's revert to code: The original code uses V as from np.linalg.svd.
        // So let's store V_ as svd.matrixV() (which is V_eigen) transposed to match python's shape:
        // Python V is (features,features), so after transpose we have same shape, but we have V_ = V_eigen^T.
        // Wait we must be consistent:
        // Python: U (m x n), s (min(m,n)), V (n x n) so Xp (m x n), U(m x n), s(n), V(n x n)
        // For full_matrices=False: U is (m x r), V is (r x n). Actually let's trust the code: 
        // The code sets full_matrices=False, so shapes:
        // For X(m x n):
        // U: (m x r), V: (r x n) 
        // With m≥n: r = n
        // Let's just store V_ as given by Eigen as matrixV(), no transpose:
        // Because python code uses V (from svd) directly. python returns V as (n x n), 
        // Eigen returns V also as (n x n) but in a form that Xp = U * S * V^T.
        // Python: Xp = U * diag(s) * V
        // If we want the same V as python: we must store V_ = V from eigen but transposed:
        // since eigen's decomposition is Xp = U*S*V^T, python's is Xp = U*S*V,
        // so python's V = eigen's V^T.
        // So we do: space_mapper.V_ = svd.matrixV(); // This would be V_eigen
        // But we want python's V:
        // python V = (eigen V)^T
        // do: space_mapper.V_ = svd.matrixV();
        // Actually we want exactly python's behavior. The python code: "U, s, V = ...", note that python's V is what's returned by np.linalg.svd.
        // np.linalg.svd returns V already so that X = U * S * V. 
        // Eigen returns V so that X = U * S * V^T.
        // So the V from python is actually V_eigen^T.
        // We must therefore do:
        // space_mapper.V_ = svd.matrixV().transpose();
        
        // We did that above. Good.

        if (dimension == -1 && threshold == 0.0) {
            threshold = 1e-6;
        }

        int stop_s;
        if (threshold != 0.0 && dimension == -1) {
            // threshold-based
            // replicate the search:
            const Eigen::VectorXd &s = space_mapper.s_;
            std::vector<double> s_rev(s.size());
            for (int i=0; i<s.size(); i++) s_rev[i] = s(s.size()-1 - i);
            auto it = std::lower_bound(s_rev.begin(), s_rev.end(), threshold);
            int idx = (int)(it - s_rev.begin());
            stop_s = (int)s.size() - idx;
        } else if (dimension != -1) {
            stop_s = dimension;
        } else {
            // default threshold if dimension not given
            const Eigen::VectorXd &s = space_mapper.s_;
            std::vector<double> s_rev(s.size());
            for (int i=0; i<s.size(); i++) s_rev[i] = s(s.size()-1 - i);
            auto it = std::lower_bound(s_rev.begin(), s_rev.end(), threshold);
            int idx = (int)(it - s_rev.begin());
            stop_s = (int)s.size() - idx;
        }

        space_mapper.stop_s = stop_s;

        if (enable_scale) {
            // scale = [1./(max(x)-min(x)) for x in np.dot(Xp, V[:stop_s].T).T]
            // Let's compute projection: Xp * V[:stop_s].T = shape (m, stop_s)
            // In Eigen: V[:stop_s] is V_.topRows(stop_s) shape (stop_s, n)
            // (Xp) is (m,n)
            // Xp * V[:stop_s].transpose() => Xp * V_.topRows(stop_s).transpose() = (m,n)*(n,stop_s)=(m, stop_s)
            
            Eigen::MatrixXd proj = Xp * space_mapper.V_.topRows(stop_s).transpose(); 
            // proj: (m, stop_s)
            
            // For each dimension (column) of proj, find max and min
            space_mapper.scale.resize(stop_s);
            for (int i=0; i<stop_s; i++) {
                double cmin = proj.col(i).minCoeff();
                double cmax = proj.col(i).maxCoeff();
                double denom = cmax - cmin;
                // if denom is 0, scale might be inf. Python code doesn't handle that explicitly.
                // We'll assume no degenerate dimension. If it occurs, we might set scale(i)=1.
                if (denom == 0.0) {
                    space_mapper.scale(i) = 1.0; 
                } else {
                    space_mapper.scale(i) = 1.0 / denom;
                }
            }
        } else {
            space_mapper.scale.resize(0);
        }

        return space_mapper;
    }
};

