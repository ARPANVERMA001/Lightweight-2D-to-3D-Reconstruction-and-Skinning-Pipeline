#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "MeshLoader.h"
#include <vector>

class Visualization {
public:
    Visualization();
    ~Visualization();
    GLuint boneVAO;  // For storing bone vertex attributes
    GLuint boneVBO; 
    bool initialize();
    void render(const MeshLoader& loader,std::vector<std::vector<float>> skinningWeights,
    std::vector<std::vector<Eigen::Matrix4f>> boneTransforms);
    bool shouldClose() const { return glfwWindowShouldClose(window); }

private:
    GLFWwindow* window;
    GLuint shaderProgram;
    GLuint VBO[2], VAO[2], EBO[2];  // Arrays for original and computed mesh
    bool buffersInitialized;
    GLuint normalVBO[2];  // Buffer for vertex normals
    const MeshLoader* meshLoader; 
    void setupShaders();
    void setupBuffers(const MeshLoader& loader);
    void updateMesh(const std::vector<Eigen::Vector3f>& vertices, int meshIndex,const MeshLoader& loader);
    void updateBonePositions(
                                      const std::vector<std::vector<float>>& skinningWeights,
                                      const std::vector<Eigen::Vector3f>& vertices);
    void normalizeVertices(std::vector<float>& vertices);
    void renderText(const char* text, float x, float y);
    std::vector<Eigen::Vector3f> computeDeformedVertices(
    const std::vector<Eigen::Vector3f>& restPose,
    const std::vector<std::vector<float>>& weights,
    const std::vector<Eigen::Matrix4f>& transforms);
};