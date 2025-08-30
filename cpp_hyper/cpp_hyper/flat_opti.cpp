#include <Eigen/Dense>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include "TriMesh.h"
#include "space_mapper.cpp"
#include "flat_intersection_biquadratic.h"
#include "format_loader.cpp"

namespace fs = std::filesystem;
void optimize_biquadratic(P,H rest_vs,deformed_vs,x0){
    auto normalization=normalization_factor
}
// Helper function to glob files matching pattern
std::vector<fs::path> glob(const std::string& pattern) {
    std::vector<fs::path> paths;
    fs::path dir = fs::path(pattern).parent_path();
    std::string ext = fs::path(pattern).extension().string();
    std::string stem = fs::path(pattern).stem().string();
    stem = stem.substr(0, stem.find('*'));
    
    for(const auto& entry : fs::directory_iterator(dir)) {
        if(entry.path().extension() == ext && 
           entry.path().stem().string().find(stem) == 0) {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

// Helper function to pad number with zeros
std::string pad_number(int num, int width) {
    std::string s = std::to_string(num);
    return std::string(width - s.length(), '0') + s;
}

int main() {
    try {
        // Get OBJ name
        fs::path obj_path = "./PerVertex/cube.obj";
        std::string obj_name = obj_path.stem().string();
        std::cout << "The name for the OBJ is: " << obj_name << std::endl;

        // Load rest mesh
        TriMesh rest_mesh;
        rest_mesh.ReadOBJ(obj_path.string());
        Eigen::MatrixXd rest_vs = rest_mesh.GetVertices();
        std::cout << "rest_vs.shape: " << rest_vs.rows() << "x" << rest_vs.cols() << std::endl;
        Eigen::MatrixXd rest_vs_original = rest_vs;

        // Get pose paths
        std::vector<fs::path> pose_paths = glob("./PerVertex/poses-1/cube-*.obj");
        fs::path pose_dir = "/PerVertex/poses-1";
        std::string pose_name = pose_dir.filename().string();
        std::cout << "The name for pose folder is: " << pose_name << std::endl;

        // Load deformed meshes
        std::vector<Eigen::MatrixXd> deformed_meshes;
        for(const auto& path : pose_paths) {
            TriMesh mesh;
            mesh.ReadOBJ(path.string());
            deformed_meshes.push_back(mesh.GetVertices());
        }

        // Convert to single matrix and reshape
        int P = deformed_meshes.size();
        int N = deformed_meshes[0].rows();
        
        // Create deformed_vs with shape (N, P, 3)
        Eigen::MatrixXd deformed_vs(N, P * 3);
        for(int i = 0; i < P; i++) {
            deformed_vs.block(0, i*3, N, 3) = deformed_meshes[i];
        }
        Eigen::MatrixXd deformed_vs_original = deformed_vs;

        // Load qs data
        Eigen::MatrixXd qs_data;
        std::ifstream qs_file("./PerVertex/qs.txt");
        if(!qs_file) {
            throw std::runtime_error("Cannot open qs.txt");
        }
        
        // Count lines and columns
        std::string line;
        int rows = 0, cols = 0;
        std::getline(qs_file, line);
        std::stringstream ss(line);
        double val;
        while(ss >> val) cols++;
        rows++;
        while(std::getline(qs_file, line)) rows++;
        
        // Reset file and read data
        qs_file.clear();
        qs_file.seekg(0);
        qs_data.resize(rows, cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                qs_file >> qs_data(i,j);
            }
        }
        std::cout << "# of good valid vertices: " << qs_data.rows() << std::endl;

        // PCA and initial guess
        const int H = 4;
        SpaceMapper pca = SpaceMapper::Uncorrellated_Space(qs_data, true, 0.0, H);
        Eigen::VectorXd pt = pca.Xavg_;
        Eigen::MatrixXd B = pca.V_.topRows(H-1).transpose();
        
        Eigen::VectorXd x0 = FlatIntersectionBiquadratic::pack(pt, B);

        // Optimize
        auto [converged, F_list, w_list] = FlatIntersectionBiquadratic::optimize_biquadratic(
            P, H, rest_vs, deformed_vs, x0);
        std::cout << "Converged: " << converged << std::endl;

        // Compute Fw_matrix
        Eigen::MatrixXd Fw_matrix(w_list.size(), F_list.cols());
        for(size_t i = 0; i < w_list.size(); i++) {
            Fw_matrix.row(i) = F_list * w_list[i];
        }
        std::cout << "FW: " << Fw_matrix.rows() << "x" << Fw_matrix.cols() << std::endl;

        // Plot results
        plotting_flat(Fw_matrix);

        // Save transformations
        int poses = Fw_matrix.cols() / 12;
        for(int i = 0; i < poses; i++) {
            std::string name = pad_number(i+1, 4);
            
            Eigen::MatrixXd per_pose_transform = Fw_matrix.block(0, i*12, Fw_matrix.rows(), 12);
            
            // Reshape to (N, 3, 4)
            Eigen::MatrixXd reshaped = Eigen::Map<Eigen::MatrixXd>(
                per_pose_transform.data(),
                per_pose_transform.rows(),
                3,
                4
            );
            
            // Transpose and reshape to (N, 12)
            Eigen::MatrixXd final_transform = Eigen::Map<Eigen::MatrixXd>(
                reshaped.transpose().data(),
                per_pose_transform.rows(),
                12
            );
            
            std::string output_path = "./cubeOut/" + name + ".DMAT";
            format_loader::write_DMAT(output_path, final_transform);
        }

    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}