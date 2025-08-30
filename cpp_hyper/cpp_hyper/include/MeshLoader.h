// include/MeshLoader.h
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>

class MeshLoader {
public:
    bool loadRestPose(const std::string& filePath);
    bool loadDeformedPoses(const std::vector<std::string>& filePaths);

    // Getters
    const std::vector<Eigen::Vector3f>& getRestVertices() const { return restVertices; }
    const std::vector<std::vector<int>>& getFaces() const { return faces; }
    const std::vector<std::vector<Eigen::Vector3f>>& getDeformedVertices() const { return deformedVertices; }
    const std::vector<std::vector<int>>& getVertexNeighbors() const { return vertexNeighbors; }

private:
    bool loadOBJ(const std::string& filePath, std::vector<Eigen::Vector3f>& vertices, std::vector<std::vector<int>>& faces);
    void buildVertexNeighborhoods();

    std::vector<Eigen::Vector3f> restVertices;
    std::vector<std::vector<int>> faces;
    std::vector<std::vector<Eigen::Vector3f>> deformedVertices; // Per pose vertices
    std::vector<std::vector<int>> vertexNeighbors; // One-ring neighbors for each vertex
};