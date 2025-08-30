#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <cmath>
#include <Eigen/Dense>

// Include the previously implemented headers
#include "format_loader.h"
#include "trimesh.h"
#include "SpaceMapper.h"

// Placeholder for QP solver
// In practice, integrate a C++ QP solver like qpOASES, OSQP, etc.
namespace mves2 {
    /**
     * Placeholder MVES function.
     * @param data The input data matrix (num_points x dimension).
     * @param method The optimization method (unused in this stub).
     * @return A tuple containing the solution matrix, weights matrix, and iteration count.
     */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, int> MVES(const Eigen::MatrixXd &data, const std::string &method="qp") {
        // Implementing a stub: return identity-based simplex
        int dim = data.cols();
        Eigen::MatrixXd solution = Eigen::MatrixXd::Identity(dim+1, dim+1);
        Eigen::MatrixXd weights = Eigen::MatrixXd::Ones(data.rows(), 1);
        int iter_num = 1;
        return std::make_tuple(solution, weights, iter_num);
    }
}

int main(int argc, char** argv){
    // Ensure correct usage
    if(argc < 2){
        std::cerr << "Usage: " << argv[0] << " <DMAT_FILE>" << std::endl;
        return -1;
    }

    std::string per_vertex_folder = "./cubeOut";
    std::string dmat_file = per_vertex_folder + "/some_input.DMAT";

    // Load DMAT using format_loader
    Eigen::MatrixXd Ts;
    try{
        Ts = load_DMAT(dmat_file).transpose();
    }
    catch(const std::exception &e){
        std::cerr << "Error loading DMAT: " << e.what() << std::endl;
        return -1;
    }
    std::cout << "Ts shape: " << Ts.rows() << "x" << Ts.cols() << std::endl;

    // Define H (dimension for SpaceMapper)
    int H=4;
    // Perform Uncorrellated_Space (like PCA)
    SpaceMapper sm = SpaceMapper::Uncorrellated_Space(Ts, true, 1e-6, H);

    // Project data
    Eigen::MatrixXd uncorrelated = sm.project(Ts);
    std::cout << "uncorrelated data shape: " << uncorrelated.rows() << "x" << uncorrelated.cols() << std::endl;

    // Call MVES solver
    Eigen::MatrixXd solution, weights;
    int iter_num;
    std::tie(solution, weights, iter_num) = mves2::MVES(uncorrelated, "qp");

    // Compute volume
    double volume = std::abs(solution.determinant());
    std::cout << "Volume: " << volume << std::endl;

    // Running time placeholder
    double running_time = 0.0;

    // Recover data
    Eigen::MatrixXd recovered = sm.unproject(solution.topRows(solution.rows()-1).transpose());

    // Define output paths
    std::string output_path = per_vertex_folder + "/result.txt";
    std::string weights_path = per_vertex_folder + "/weights.txt";

    // Write result using format_loader
    try{
        write_result(output_path, weights_path, recovered, weights, iter_num, running_time, true);
    }
    catch(const std::exception &e){
        std::cerr << "Error writing results: " << e.what() << std::endl;
        return -1;
    }

    // Load a mesh using TriMesh
    TriMesh rest_mesh;
    try{
        rest_mesh = TriMesh::FromOBJ_FileName("./cube.obj");
    }
    catch(const std::exception &e){
        std::cerr << "Error loading mesh: " << e.what() << std::endl;
        return -1;
    }
    std::cout << "Loaded mesh with " << rest_mesh.vs.size() << " vertices." << std::endl;

    return 0;
}