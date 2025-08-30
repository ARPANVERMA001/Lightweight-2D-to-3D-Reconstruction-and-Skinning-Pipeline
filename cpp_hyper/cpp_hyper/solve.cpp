

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <string>
#include <sstream>
#include <stdexcept>
#include <random>
#include <chrono>
#include <filesystem>
#include <numeric>
#include <iomanip> // For setprecision

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/LU>

class TriMesh {
public:
    std::vector<std::array<double,3>> vs;
    std::vector<std::array<int,3>> faces;

    static TriMesh FromOBJ_FileName(const std::string& fname) {
        std::ifstream f(fname.c_str());
        if(!f.is_open()) {
            throw std::runtime_error("Cannot open OBJ file");
        }
        return FromOBJ_Lines(f);
    }

    static TriMesh FromOBJ_Lines(std::istream& in) {
        TriMesh result;
        std::string line;
        while(std::getline(in,line)) {
            std::stringstream ss(line);
            std::string token;
            ss >> token;
            if(!ss) continue;
            if(token=="v") {
                double x,y,z;
                ss >> x >> y >> z;
                result.vs.push_back({x,y,z});
            } else if(token=="f") {
                std::vector<int> face_indices;
                std::string vstr;
                while(ss >> vstr) {
                    int slash_pos = (int)vstr.find('/');
                    std::string v_idx_str = (slash_pos == (int)std::string::npos) ? vstr : vstr.substr(0,slash_pos);
                    int idx = std::stoi(v_idx_str);
                    if(idx<0) {
                        idx = (int)result.vs.size() + idx;
                    } else {
                        idx = idx-1;
                    }
                    face_indices.push_back(idx);
                }
                if(face_indices.size()==3) {
                    result.faces.push_back({face_indices[0], face_indices[1], face_indices[2]});
                }
            }
        }
        // Validate faces
        for (auto &f : result.faces) {
            for (int i=0; i<3; i++) {
                if(f[i]<0 || f[i]>=(int)result.vs.size())
                    throw std::runtime_error("Invalid face index in OBJ");
            }
        }
        return result;
    }
};

// Implement Python-like pinv: similar to np.linalg.pinv(A)
static Eigen::MatrixXd pinv(const Eigen::MatrixXd &A) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU|Eigen::ComputeThinV);
    Eigen::VectorXd S = svd.singularValues();
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();
    // Python's np.linalg.pinv default tolerance:
    // tol = max(A.shape)*np.finfo(S.dtype).eps * S[0]
    // We approximate by: tol = S[0]*1e-15 (common default)
    double tol = S(0)*1e-15;
    Eigen::VectorXd S_inv = S;
    for (int i=0; i<S.size(); i++) {
        if (S(i)>tol) S_inv(i)=1.0/S(i); else S_inv(i)=0.0;
    }
    return V * S_inv.asDiagonal() * U.transpose();
}

// smallest singular value, equivalent to np.linalg.norm(M, ord=-2)
static double smallest_singular_value(const Eigen::MatrixXd &M) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M);
    return svd.singularValues().minCoeff();
}

static double find_scale(const Eigen::MatrixXd &Vertices) {
    Eigen::VectorXd Dmin = Vertices.colwise().minCoeff();
    Eigen::VectorXd Dmax = Vertices.colwise().maxCoeff();
    Eigen::VectorXd D = Dmax - Dmin;
    double scale = std::sqrt(D.dot(D));
    return scale;
}

// Python-like rounding: .round() uses round half to even
static int py_round_to_int(double val) {
    double floor_val = std::floor(val);
    double diff = val - floor_val;
    if (diff > 0.5) {
        return (int)(floor_val+1);
    } else if (diff < 0.5) {
        return (int)floor_val;
    } else {
        // exactly 0.5
        // half to even
        int f_int = (int)floor_val;
        // if odd, round up
        if ((f_int % 2) != 0) {
            return f_int+1;
        } else {
            return f_int;
        }
    }
}

// Solve directly function
static std::tuple<Eigen::VectorXd,double,double> solve_directly(
    const Eigen::MatrixXd &V0,
    const Eigen::MatrixXd &V1,
    bool use_pseudoinverse = false
) {
    int pose_num = (int)V1.cols()/3;
    Eigen::MatrixXd left = Eigen::MatrixXd::Zero(4,4);
    Eigen::MatrixXd right = Eigen::MatrixXd::Zero(4,3*pose_num);
    double constant=0.0;

    // Accumulate left, right, constant
    for (int i=0; i<V0.rows(); i++) {
        Eigen::VectorXd v0 = V0.row(i).transpose(); 
        Eigen::VectorXd v1 = V1.row(i).transpose(); 
        left += v0*v0.transpose();
        right += v0*v1.transpose();
        constant += v1.dot(v1);
    }

    double ssv = smallest_singular_value(left);
    if (ssv < 1e-10) use_pseudoinverse = true;

    Eigen::RowVectorXd v0_center = V0.row(0);
    Eigen::RowVectorXd v1_center = V1.row(0);

    Eigen::MatrixXd new_left = Eigen::MatrixXd::Zero(5,5);
    new_left.topLeftCorner(4,4) = left;
    for (int r=0; r<4; r++) {
        new_left(r,4)=v0_center(r);
        new_left(4,r)=v0_center(r);
    }

    Eigen::MatrixXd new_right(5,3*pose_num);
    new_right.topRows(4)=right;
    new_right.row(4)=v1_center;

    Eigen::MatrixXd x_full;
    if (use_pseudoinverse) {
        x_full = pinv(new_left)*new_right;
    } else {
        // Equivalent to scipy.linalg.solve for a general square system
        // uses LU decomposition with partial pivoting (dgesv)
        x_full = new_left.partialPivLu().solve(new_right);
    }

    // x = x_full[:-1].ravel(order='F')
    Eigen::MatrixXd X_cut = x_full.topRows(4); 
    Eigen::VectorXd x(X_cut.size());
    {
        int idx=0;
        for (int col=0; col<X_cut.cols(); col++) {
            for (int row=0; row<X_cut.rows(); row++) {
                x(idx++) = X_cut(row,col);
            }
        }
    }

    // y = x.reshape((4,-1), order='F')
    Eigen::MatrixXd y(4,3*pose_num);
    {
        int idx=0;
        for (int col=0; col<3*pose_num; col++) {
            for (int row=0; row<4; row++) {
                y(row,col)=x(idx++);
            }
        }
    }

    // cost = (y*(left*y)).sum() - 2*(right*y).sum() + constant
    Eigen::MatrixXd Ly = left*y;
    double part1=0.0;
    for (int rr=0; rr<y.rows(); rr++) {
        for (int cc=0; cc<y.cols(); cc++) {
            part1 += y(rr,cc)*Ly(rr,cc);
        }
    }
    double part2=0.0;
    for (int rr=0; rr<right.rows(); rr++) {
        for (int cc=0; cc<right.cols(); cc++) {
            part2 += right(rr,cc)*y(rr,cc);
        }
    }
    double cost = part1 - 2*part2 + constant;

    return std::make_tuple(x,cost,ssv);
}

// find_subspace_intersections function
static std::tuple<Eigen::MatrixXd,Eigen::VectorXd,Eigen::VectorXd>
find_subspace_intersections(
    const TriMesh &rest_pose,
    const std::vector<TriMesh> &other_poses,
    int version=1,
    const std::string &method="vertex",
    bool use_pseudoinverse=false,
    bool propagate=false,
    const std::string &random_sample_close_vertex="euclidean",
    int candidate_num=120, int sample_num=10, int vertices_num=30,
    const Eigen::MatrixXd *precomputed_geodesic_distance=nullptr
) {
    std::cout << use_pseudoinverse << " " << propagate << " " << method << "\n";
    int pose_num=(int)other_poses.size();

    Eigen::MatrixXd vertices0(rest_pose.vs.size(),4);
    for (int i=0; i<(int)rest_pose.vs.size(); i++) {
        vertices0(i,0)=rest_pose.vs[i][0];
        vertices0(i,1)=rest_pose.vs[i][1];
        vertices0(i,2)=rest_pose.vs[i][2];
        vertices0(i,3)=1.0;
    }

    Eigen::MatrixXd vertices1(rest_pose.vs.size(), 3*pose_num);
    for (int p=0; p<pose_num; p++) {
        for (int i=0; i<(int)rest_pose.vs.size(); i++) {
            vertices1(i,3*p+0)=other_poses[p].vs[i][0];
            vertices1(i,3*p+1)=other_poses[p].vs[i][1];
            vertices1(i,3*p+2)=other_poses[p].vs[i][2];
        }
    }

    int n=(int)rest_pose.vs.size();
    Eigen::MatrixXd data = vertices0.leftCols(3);
    Eigen::MatrixXd vertices_pairwise_distance(n,n);
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            double dx=data(i,0)-data(j,0);
            double dy=data(i,1)-data(j,1);
            double dz=data(i,2)-data(j,2);
            vertices_pairwise_distance(i,j)=std::sqrt(dx*dx+dy*dy+dz*dz);
        }
    }

    double scale = find_scale(vertices0.leftCols(3));

    // np.random.seed(1)
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> uniform_dist(0.0,1.0);

    std::vector<Eigen::VectorXd> q_space_list;
    q_space_list.reserve(n);
    std::vector<double> errors_list; errors_list.reserve(n);
    std::vector<double> ssv_list; ssv_list.reserve(n);

    for (int i=0; i<n; i++) {
        std::vector<std::pair<double,int>> dinds;
        dinds.reserve(n);
        for (int j=0; j<n; j++) {
            dinds.push_back({vertices_pairwise_distance(i,j),j});
        }
        std::sort(dinds.begin(), dinds.end(), [](auto &a,auto &b){return a.first<b.first;});

        double best_error = std::numeric_limits<double>::infinity();
        Eigen::VectorXd best_q;
        double best_ssv = 0.0;

        for (int sample=0; sample<sample_num; sample++) {
            std::vector<int> random_indices;
            random_indices.reserve(vertices_num);
            for (int vv=0; vv<vertices_num; vv++) {
                double r = uniform_dist(rng)*candidate_num;
                int idx = py_round_to_int(r);
                if (idx>=candidate_num) idx=candidate_num-1;
                random_indices.push_back(dinds[idx].second);
            }

            Eigen::MatrixXd V0(1+(int)random_indices.size(),4);
            V0.row(0)=vertices0.row(i);
            for (size_t k=0; k<random_indices.size(); k++) {
                V0.row((int)k+1)=vertices0.row(random_indices[k]);
            }

            Eigen::MatrixXd V1(1+(int)random_indices.size(),3*pose_num);
            V1.row(0)=vertices1.row(i);
            for (size_t k=0; k<random_indices.size(); k++) {
                V1.row((int)k+1)=vertices1.row(random_indices[k]);
            }

            auto [x,cost,ssv] = solve_directly(V0,V1,use_pseudoinverse);
            double val = cost/(pose_num*scale*scale);
            double err = std::sqrt(std::max(val,1e-30));
            if (err<best_error) {
                best_error=err;
                best_q=x;
                best_ssv=ssv;
            }
        }

        q_space_list.push_back(best_q);
        errors_list.push_back(best_error);
        ssv_list.push_back(best_ssv);
    }

    Eigen::MatrixXd q_space(n, (int)q_space_list[0].size());
    for (int ii=0; ii<n; ii++) {
        q_space.row(ii)=q_space_list[ii].transpose();
    }

    Eigen::VectorXd errors(n);
    for (int ii=0; ii<n; ii++) errors(ii)=errors_list[ii];

    Eigen::VectorXd smallest_singular_values(n);
    for (int ii=0; ii<n; ii++) smallest_singular_values(ii)=ssv_list[ii];

    assert((int)q_space.rows() == n);
    return std::make_tuple(q_space, errors, smallest_singular_values);
}

static void save_one(const std::string &path, const Eigen::MatrixXd &M) {
    std::ofstream out(path);
    if(!out.is_open()) {
        std::cerr << "Cannot open file to write: " << path << "\n";
        return;
    }
    // Match np.savetxt format
    out << std::scientific << std::setprecision(18);
    for (int i=0; i<M.rows(); i++) {
        for (int j=0; j<M.cols(); j++) {
            out << M(i,j);
            if(j+1<M.cols()) out << " ";
        }
        out << "\n";
    }
    out.close();
    std::cout << "Saved to path: " << path << "\n";
}

int main() {
    std::string rest_pose_path="./cat/0001.obj";
    std::string other_poses_path="./cat/poses";
    std::cout << "Loading rest pose mesh: " << rest_pose_path << "\n";
    TriMesh rest_pose = TriMesh::FromOBJ_FileName(rest_pose_path);

    std::filesystem::path directory_path(other_poses_path);
    std::vector<std::string> obj_files;
    for (auto &entry : std::filesystem::directory_iterator(directory_path)) {
        if(entry.is_regular_file()) {
            if(entry.path().extension()==".obj") {
                obj_files.push_back(entry.path().filename().string());
            }
        }
    }
    std::sort(obj_files.begin(), obj_files.end());

    std::cout << "[";
    for (auto &fn: obj_files) std::cout << fn << " ";
    std::cout << "]\n";
    std::cout << "Loading " << other_poses_path.size() << " other mesh poses..." << "\n";

    std::vector<TriMesh> other_poses;
    for (auto &path : obj_files) {
        TriMesh m = TriMesh::FromOBJ_FileName(other_poses_path+"/"+path);
        other_poses.push_back(m);
    }

    std::cout << (int)other_poses.size() << "\n";
    std::cout << "...done.\n";

    std::cout << "Generating transformations...\n";

    auto start_time = std::chrono::steady_clock::now();
    auto [qs, errors, smallest_singular_values] = find_subspace_intersections(rest_pose, other_poses);
    auto end_time = std::chrono::steady_clock::now();
    double dur = std::chrono::duration<double>(end_time - start_time).count()/60.0;
    std::cout << "... Finished generating transformations.\n";
    std::cout << "Finding subspace intersection duration (minutes): " << dur << "\n";

    std::cout << "qs shape: " << qs.rows() << " " << qs.cols() << "\n";
    std::cout << rest_pose.vs.size() << " " << 3 << "\n";
    std::cout << smallest_singular_values.transpose() << "\n";

    Eigen::MatrixXd vs_mat(rest_pose.vs.size(),3);
    for (int i=0; i<(int)rest_pose.vs.size(); i++) {
        vs_mat(i,0)=rest_pose.vs[i][0];
        vs_mat(i,1)=rest_pose.vs[i][1];
        vs_mat(i,2)=rest_pose.vs[i][2];
    }
    std::cout << find_scale(vs_mat) << "\n";
    std::cout << qs.rows() << " " << qs.cols() << "\n";

    save_one("./qs.txt", qs);

    return 0;
}