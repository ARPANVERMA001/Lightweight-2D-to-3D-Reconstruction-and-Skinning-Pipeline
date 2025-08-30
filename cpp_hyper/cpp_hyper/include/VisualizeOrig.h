#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "MeshLoader.h"
#include "SkinningSolver.h"
#include <vector>

class VisualizeOrig {
public:
    VisualizeOrig();
    ~VisualizeOrig();

    bool initialize();
    void render(const MeshLoader& loader);
    bool shouldClose() const { return glfwWindowShouldClose(window); }

private:
    GLFWwindow* window;
    GLuint shaderProgram;
    GLuint VBO, VAO, EBO;
    bool buffersInitialized;
    
    void setupShaders();
    void setupBuffers(const MeshLoader& loader);
    void updateMesh(const std::vector<Eigen::Vector3f>& vertices);
    void normalizeVertices(std::vector<float>& vertices);
};
