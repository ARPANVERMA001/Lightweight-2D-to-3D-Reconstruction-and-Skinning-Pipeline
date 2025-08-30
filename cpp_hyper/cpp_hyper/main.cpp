// main.cpp
#include "include/MeshLoader.h"
// #include "include/SkinningSolver.h"
#include "include/Visualization.h"
// #include "include/VisualizeOrig.h"

#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <chrono>

namespace fs = std::filesystem;
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>

std::vector<std::vector<float>> readWeights(const std::string& filePath) {
    std::vector<std::vector<float>> weights;
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open weights file: " + filePath);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        std::vector<float> vertexWeights;
        float weight;
        while (stream >> weight) {
            vertexWeights.push_back(weight);
        }
        weights.push_back(vertexWeights);
    }

    return weights;
}

std::vector<std::vector<Eigen::Matrix4f>> readTransforms(const std::string& filePath) {
    std::vector<std::vector<Eigen::Matrix4f>> transforms;
    std::ifstream file(filePath);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open transforms file: " + filePath);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        std::vector<Eigen::Matrix4f> lineTransforms;
        
        // Read 4 matrices for this line
        for (int matrixIdx = 0; matrixIdx < 4; matrixIdx++) {
            Eigen::Matrix4f transform = Eigen::Matrix4f::Zero();
            
            // For each matrix, read 3 rows (3x4 matrix)
            for (int row = 0; row < 3; row++) {
                for (int col = 0; col < 4; col++) {
                    stream >> transform(row, col);
                }
            }
            transform(3, 0) = 0;
            transform(3, 1) = 0;
            transform(3, 2) = 0;
            transform(3, 3) = 1;
            
            lineTransforms.push_back(transform);
        }
        
        transforms.push_back(lineTransforms);
    }
    
    return transforms;
}

void printProgress(const std::string& stage) {
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count();
    
    std::cout << "[" << duration << "s] " << stage << std::endl;
}

int main() {
    try {
        printProgress("Starting skinning computation...");
        
        // Initialize mesh loader
        MeshLoader loader;
        
        // Load rest pose
        printProgress("Loading rest pose...");
        std::string restPosePath = "cat/0001.obj";
        if (!loader.loadRestPose(restPosePath)) {
            std::cerr << "Failed to load rest pose from " << restPosePath << std::endl;
            return 1;
        }
        printProgress("Rest pose loaded successfully");
        
        // Load all pose files from the assets/cat-poses directory
        printProgress("Loading pose files...");
        std::vector<std::string> posePaths;
        try {
            for (const auto& entry : fs::directory_iterator("cat/poses")) {
                if (entry.path().extension() == ".obj") {
                    posePaths.push_back(entry.path().string());
                }
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Error accessing directory: " << e.what() << std::endl;
            return 1;
        }
        
        // Check if we found any pose files
        if (posePaths.empty()) {
            std::cerr << "No pose files found in assets/cat-poses!" << std::endl;
            return 1;
        }
        printProgress("Found " + std::to_string(posePaths.size()) + " pose files");
        
        // Sort the paths to ensure consistent ordering
        std::sort(posePaths.begin(), posePaths.end());
        
        // Load the deformed poses
        if (!loader.loadDeformedPoses(posePaths)) {
            std::cerr << "Failed to load deformed poses!" << std::endl;
            return 1;
        }
        printProgress("All poses loaded successfully");
        
        // Initialize the skinning solver
        printProgress("Initializing skinning solver...");
        // SkinningSolver solver;
        // solver.setNumBones(10); // Set number of bones - adjust as needed
        
        // Compute skinning
        printProgress("Computing skinning...");
        // solver.computeSkinning(loader);
        
        // Print some statistics
        
        // If you have visualization code:
        // Visualization visualization;
        // if(! visualization.initialize()){
        //     std::cerr<<"Failed to initialize visualization"<<std::endl;
        //     return 1;
        // }
        // while(!visualization.shouldClose()){
        //     visualization.processInput();
        //     visualization.render(loader,solver);
        // }
        std::vector<std::vector<float>> skinningWeights=readWeights("weights.txt");
        std::vector<std::vector<Eigen::Matrix4f>> boneTransform=readTransforms("transforms.txt");
        Visualization visualization;
        if(! visualization.initialize()){
            std::cerr<<"Failed to initialize visualization"<<std::endl;
            return 1;
        }
        while(!visualization.shouldClose()){
            // visualization.processInput();
            // std::cout<<"Rendering"<<std::endl;
            visualization.render(loader,skinningWeights,boneTransform);
        }
        // visualization.render(loader, solver)
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred!" << std::endl;
        return 1;
    }
}