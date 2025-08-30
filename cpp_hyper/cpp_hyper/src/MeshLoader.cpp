// src/MeshLoader.cpp
#include "../include/MeshLoader.h"
#include <tiny_obj_loader.h>
#include <iostream>

bool MeshLoader::loadRestPose(const std::string& filePath) {
    bool success = loadOBJ(filePath, restVertices, faces);
    if(success) {
        buildVertexNeighborhoods();
    }
    return success;
}

bool MeshLoader::loadDeformedPoses(const std::vector<std::string>& filePaths) {
    deformedVertices.clear();
    deformedVertices.reserve(filePaths.size());
    
    for(const auto& path : filePaths) {
        std::vector<Eigen::Vector3f> vertices;
        std::vector<std::vector<int>> tempFaces;
        
        if(!loadOBJ(path, vertices, tempFaces)) {
            return false;
        }
        
        if(vertices.size() != restVertices.size()) {
            std::cerr << "Pose mesh vertex count mismatch!" << std::endl;
            return false;
        }
        
        deformedVertices.push_back(std::move(vertices));
    }
    
    return true;
}

bool MeshLoader::loadOBJ(const std::string& filePath, 
                        std::vector<Eigen::Vector3f>& vertices, 
                        std::vector<std::vector<int>>& faces) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool success = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, 
                                  filePath.c_str());
    
    if (!warn.empty()) {
        std::cout << "Warning: " << warn << std::endl;
    }
    
    if (!err.empty()) {
        std::cerr << "Error: " << err << std::endl;
    }

    if (!success) {
        return false;
    }

    // Convert vertices
    vertices.clear();
    vertices.reserve(attrib.vertices.size() / 3);
    for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
        vertices.emplace_back(attrib.vertices[i], 
                            attrib.vertices[i + 1],
                            attrib.vertices[i + 2]);
    }

    // Convert faces
    faces.clear();
    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            std::vector<int> face;
            int fv = shape.mesh.num_face_vertices[f];
            for (int v = 0; v < fv; v++) {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                face.push_back(idx.vertex_index);
            }
            faces.push_back(face);
            index_offset += fv;
        }
    }

    return true;
}

void MeshLoader::buildVertexNeighborhoods() {
    vertexNeighbors.clear();
    vertexNeighbors.resize(restVertices.size());
    
    // Build one-ring neighborhoods from faces
    for(const auto& face : faces) {
        for(size_t i = 0; i < face.size(); ++i) {
            int v1 = face[i];
            int v2 = face[(i + 1) % face.size()];
            vertexNeighbors[v1].push_back(v2);
            vertexNeighbors[v2].push_back(v1);
        }
    }
    
    // Remove duplicates in each neighborhood
    for(auto& neighbors : vertexNeighbors) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), 
                       neighbors.end());
    }
}