// #include "../include/VisualizeOrig.h"
// #include <iostream>

// const char* vertexShaderSource = R"(
//     #version 330 core
//     layout (location = 0) in vec3 aPos;
//     void main() {
//         gl_Position = vec4(aPos, 1.0);
//     }
// )";

// const char* fragmentShaderSource = R"(
//     #version 330 core
//     out vec4 FragColor;
//     void main() {
//         FragColor = vec4(0.8f, 0.3f, 0.2f, 1.0f);
//     }
// )";

// VisualizeOrig::VisualizeOrig() : window(nullptr), buffersInitialized(false) {}

// bool VisualizeOrig::initialize() {
//     if (!glfwInit()) {
//         std::cerr << "Failed to initialize GLFW" << std::endl;
//         return false;
//     }

//     glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//     glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//     glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

//     window = glfwCreateWindow(800, 600, "Original Mesh Visualization", NULL, NULL);
//     if (!window) {
//         std::cerr << "Failed to create GLFW window" << std::endl;
//         glfwTerminate();
//         return false;
//     }

//     glfwMakeContextCurrent(window);

//     if (glewInit() != GLEW_OK) {
//         std::cerr << "Failed to initialize GLEW" << std::endl;
//         return false;
//     }

//     setupShaders();

//     // Check shader compilation
//     GLint success;
//     GLchar infoLog[512];
//     glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
//     if(!success) {
//         glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
//         std::cerr << "Shader program linking failed: " << infoLog << std::endl;
//         return false;
//     }

//     glEnable(GL_DEPTH_TEST);
//     return true;
// }

// void VisualizeOrig::setupShaders() {
//     // Vertex shader
//     GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
//     glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
//     glCompileShader(vertexShader);

//     // Check vertex shader compilation
//     GLint success;
//     GLchar infoLog[512];
//     glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
//     if(!success) {
//         glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
//         std::cerr << "Vertex shader compilation failed: " << infoLog << std::endl;
//     }

//     // Fragment shader
//     GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
//     glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
//     glCompileShader(fragmentShader);

//     // Check fragment shader compilation
//     glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
//     if(!success) {
//         glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
//         std::cerr << "Fragment shader compilation failed: " << infoLog << std::endl;
//     }

//     // Shader program
//     shaderProgram = glCreateProgram();
//     glAttachShader(shaderProgram, vertexShader);
//     glAttachShader(shaderProgram, fragmentShader);
//     glLinkProgram(shaderProgram);

//     glDeleteShader(vertexShader);
//     glDeleteShader(fragmentShader);
// }

// void VisualizeOrig::normalizeVertices(std::vector<float>& vertices) {
//     // Find bounds
//     float minX = vertices[0], maxX = vertices[0];
//     float minY = vertices[1], maxY = vertices[1];
//     float minZ = vertices[2], maxZ = vertices[2];
    
//     for(size_t i = 0; i < vertices.size(); i += 3) {
//         minX = std::min(minX, vertices[i]);
//         maxX = std::max(maxX, vertices[i]);
//         minY = std::min(minY, vertices[i + 1]);
//         maxY = std::max(maxY, vertices[i + 1]);
//         minZ = std::min(minZ, vertices[i + 2]);
//         maxZ = std::max(maxZ, vertices[i + 2]);
//     }
    
//     // Calculate scale factor
//     float scaleX = 2.0f / (maxX - minX);
//     float scaleY = 2.0f / (maxY - minY);
//     float scale = std::min(scaleX, scaleY) * 0.8f;  // Leave some margin
    
//     // Normalize vertices
//     for(size_t i = 0; i < vertices.size(); i += 3) {
//         vertices[i] = (vertices[i] - (minX + maxX) * 0.5f) * scale;
//         vertices[i + 1] = (vertices[i + 1] - (minY + maxY) * 0.5f) * scale;
//         vertices[i + 2] = (vertices[i + 2] - (minZ + maxZ) * 0.5f) * scale;
//     }
// }

// void VisualizeOrig::setupBuffers(const MeshLoader& loader) {
//     glGenVertexArrays(1, &VAO);
//     glGenBuffers(1, &VBO);
//     glGenBuffers(1, &EBO);

//     glBindVertexArray(VAO);

//     // Prepare vertex data
//     const auto& vertices = loader.getRestVertices();
//     std::vector<float> vertexData;
//     vertexData.reserve(vertices.size() * 3);
//     for(const auto& v : vertices) {
//         vertexData.push_back(v.x());
//         vertexData.push_back(v.y());
//         vertexData.push_back(v.z());
//     }
    
//     // Normalize vertices to [-1, 1] range
//     normalizeVertices(vertexData);

//     glBindBuffer(GL_ARRAY_BUFFER, VBO);
//     glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), 
//                 vertexData.data(), GL_DYNAMIC_DRAW);

//     // Prepare index data
//     const auto& faces = loader.getFaces();
//     std::vector<unsigned int> indices;
//     indices.reserve(faces.size() * 3);
//     for(const auto& face : faces) {
//         indices.push_back(face[0]);
//         indices.push_back(face[1]);
//         indices.push_back(face[2]);
//     }

//     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
//     glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
//                 indices.data(), GL_STATIC_DRAW);

//     glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
//     glEnableVertexAttribArray(0);

//     buffersInitialized = true;
// }

// void VisualizeOrig::updateMesh(const std::vector<Eigen::Vector3f>& vertices) {
//     std::vector<float> vertexData;
//     vertexData.reserve(vertices.size() * 3);
//     for(const auto& v : vertices) {
//         vertexData.push_back(v.x());
//         vertexData.push_back(v.y());
//         vertexData.push_back(v.z());
//     }
    
//     normalizeVertices(vertexData);

//     glBindBuffer(GL_ARRAY_BUFFER, VBO);
//     glBufferSubData(GL_ARRAY_BUFFER, 0, vertexData.size() * sizeof(float), 
//                    vertexData.data());
// }

// void VisualizeOrig::render(const MeshLoader& loader) {
//     static int currentPose = 0;
//     static double lastTime = glfwGetTime();

//     if(!buffersInitialized) {
//         setupBuffers(loader);
//     }

//     // Switch poses every few seconds
//     double currentTime = glfwGetTime();
//     if(currentTime - lastTime > 2.0) {
//         currentPose = (currentPose + 1) % loader.getDeformedVertices().size();
//         lastTime = currentTime;
//         updateMesh(loader.getDeformedVertices()[currentPose]);
//     }

//     glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
//     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//     glUseProgram(shaderProgram);
//     glBindVertexArray(VAO);
    
//     // Draw mesh
//     const auto& faces = loader.getFaces();
//     glDrawElements(GL_TRIANGLES, faces.size() * 3, GL_UNSIGNED_INT, 0);

//     glfwSwapBuffers(window);
//     glfwPollEvents();
// }

// VisualizeOrig::~VisualizeOrig() {
//     if(buffersInitialized) {
//         glDeleteVertexArrays(1, &VAO);
//         glDeleteBuffers(1, &VBO);
//         glDeleteBuffers(1, &EBO);
//     }
//     glDeleteProgram(shaderProgram);
//     glfwTerminate();
// }
