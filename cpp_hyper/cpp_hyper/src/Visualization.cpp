#include "../include/Visualization.h"
#include <iostream>
#include <iomanip> 
// Modified shader sources
const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    out vec3 FragPos;
    out vec3 Normal;
    
    void main() {
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal;  // Correct normal transformation
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
)";

const char* fragmentShaderSource = R"(
    #version 330 core
    in vec3 FragPos;
    in vec3 Normal;
    
    uniform vec3 viewPos;
    uniform vec3 lightPos;
    uniform vec3 meshColor;
    
    out vec4 FragColor;
    
    void main() {
        // Light properties
        vec3 lightColor = vec3(1.0, 1.0, 1.0);
        float ambientStrength = 0.2;
        float specularStrength = 0.5;
        float shininess = 32.0;
        
        // Ambient
        vec3 ambient = ambientStrength * lightColor;
        
        // Diffuse
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;
        
        // Specular
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 halfwayDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
        vec3 specular = specularStrength * spec * lightColor;
        
        // Final color
        vec3 result = (ambient + diffuse + specular) * meshColor;
        FragColor = vec4(result, 1.0);
    }
)";

Visualization::Visualization() : window(nullptr), buffersInitialized(false), boneVAO(0), boneVBO(0)  {}

bool Visualization::initialize() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(1200, 600, "Mesh Comparison", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return false;
    }

    setupShaders();
    glEnable(GL_DEPTH_TEST);
    return true;
}

void Visualization::setupShaders() {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void Visualization::normalizeVertices(std::vector<float>& vertices) {
    if(vertices.empty()) return;
    
    float minX = vertices[0], maxX = vertices[0];
    float minY = vertices[1], maxY = vertices[1];
    float minZ = vertices[2], maxZ = vertices[2];
    
    for(size_t i = 0; i < vertices.size(); i += 3) {
        minX = std::min(minX, vertices[i]);
        maxX = std::max(maxX, vertices[i]);
        minY = std::min(minY, vertices[i + 1]);
        maxY = std::max(maxY, vertices[i + 1]);
        minZ = std::min(minZ, vertices[i + 2]);
        maxZ = std::max(maxZ, vertices[i + 2]);
    }
    
    float scaleX = 2.0f / (maxX - minX);
    float scaleY = 2.0f / (maxY - minY);
    float scale = std::min(scaleX, scaleY) * 0.8f;
    
    for(size_t i = 0; i < vertices.size(); i += 3) {
        vertices[i] = (vertices[i] - (minX + maxX) * 0.5f) * scale;
        vertices[i + 1] = (vertices[i + 1] - (minY + maxY) * 0.5f) * scale;
        vertices[i + 2] = (vertices[i + 2] - (minZ + maxZ) * 0.5f) * scale;
    }
}
void Visualization::setupBuffers(const MeshLoader& loader) {
    glGenVertexArrays(2, VAO);
    glGenBuffers(2, VBO);
    glGenBuffers(2, normalVBO);  // New buffer for normals
    glGenBuffers(2, EBO);

    const auto& vertices = loader.getRestVertices();
    const auto& faces = loader.getFaces();

    // Compute normals
    std::vector<Eigen::Vector3f> normals(vertices.size(), Eigen::Vector3f::Zero());
    for(const auto& face : faces) {
        Eigen::Vector3f v1 = vertices[face[1]] - vertices[face[0]];
        Eigen::Vector3f v2 = vertices[face[2]] - vertices[face[0]];
        Eigen::Vector3f normal = v1.cross(v2).normalized();
        
        // Add to all vertices of this face
        normals[face[0]] += normal;
        normals[face[1]] += normal;
        normals[face[2]] += normal;
    }

    // Normalize all normals
    for(auto& normal : normals) {
        normal.normalize();
    }

    // Setup for both original and computed mesh
    for(int i = 0; i < 2; i++) {
        glBindVertexArray(VAO[i]);

        // Vertex positions
        glBindBuffer(GL_ARRAY_BUFFER, VBO[i]);
        std::vector<float> vertexData;
        vertexData.reserve(vertices.size() * 3);
        for(const auto& v : vertices) {
            vertexData.push_back(v.x());
            vertexData.push_back(v.y());
            vertexData.push_back(v.z());
        }
        normalizeVertices(vertexData);
        glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float),
                    vertexData.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        // Normals
        glBindBuffer(GL_ARRAY_BUFFER, normalVBO[i]);
        std::vector<float> normalData;
        normalData.reserve(normals.size() * 3);
        for(const auto& n : normals) {
            normalData.push_back(n.x());
            normalData.push_back(n.y());
            normalData.push_back(n.z());
        }
        glBufferData(GL_ARRAY_BUFFER, normalData.size() * sizeof(float),
                    normalData.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);

        // Indices
        std::vector<unsigned int> indices;
        indices.reserve(faces.size() * 3);
        for(const auto& face : faces) {
            indices.push_back(face[0]);
            indices.push_back(face[1]);
            indices.push_back(face[2]);
        }
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[i]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
                    indices.data(), GL_STATIC_DRAW);
    }
    glGenVertexArrays(1, &boneVAO);
    glGenBuffers(1, &boneVBO);
    
    glBindVertexArray(boneVAO);
    glBindBuffer(GL_ARRAY_BUFFER, boneVBO);
    
    // Allocate space for bone positions (4 bones * 3 coordinates)
    std::vector<float> initialBonePositions(4 * 3, 0.0f);  // Adjust size based on number of bones
    glBufferData(GL_ARRAY_BUFFER, initialBonePositions.size() * sizeof(float), 
                initialBonePositions.data(), GL_DYNAMIC_DRAW);
    
    // Setup vertex attributes for bones
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    buffersInitialized = true;
}

void Visualization::updateMesh(const std::vector<Eigen::Vector3f>& vertices, int meshIndex,const MeshLoader& loader) {
    // Update vertex positions
    std::vector<float> vertexData;
    vertexData.reserve(vertices.size() * 3);
    for(const auto& v : vertices) {
        vertexData.push_back(v.x());
        vertexData.push_back(v.y());
        vertexData.push_back(v.z());
    }
    normalizeVertices(vertexData);

    glBindBuffer(GL_ARRAY_BUFFER, VBO[meshIndex]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertexData.size() * sizeof(float),
                   vertexData.data());

    // Update normals
    const auto& faces = loader.getFaces();
    std::vector<Eigen::Vector3f> normals(vertices.size(), Eigen::Vector3f::Zero());
    std::cout<<faces.size()<<std::endl;
    for(const auto& face : faces) {
        Eigen::Vector3f v1 = vertices[face[1]] - vertices[face[0]];
        Eigen::Vector3f v2 = vertices[face[2]] - vertices[face[0]];
        Eigen::Vector3f normal = v1.cross(v2).normalized();
        
        normals[face[0]] += normal;
        normals[face[1]] += normal;
        normals[face[2]] += normal;
    }

    for(auto& normal : normals) {
        normal.normalize();
    }

    std::vector<float> normalData;
    normalData.reserve(normals.size() * 3);
    for(const auto& n : normals) {
        normalData.push_back(n.x());
        normalData.push_back(n.y());
        normalData.push_back(n.z());
    }

    glBindBuffer(GL_ARRAY_BUFFER, normalVBO[meshIndex]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, normalData.size() * sizeof(float),
                   normalData.data());
}

// void Visualization::updateMesh(const std::vector<Eigen::Vector3f>& vertices, int meshIndex) {
//     std::vector<float> vertexData;
//     vertexData.reserve(vertices.size() * 3);
//     for(const auto& v : vertices) {
//         vertexData.push_back(v.x());
//         vertexData.push_back(v.y());
//         vertexData.push_back(v.z());
//     }
    
//     normalizeVertices(vertexData);

//     glBindBuffer(GL_ARRAY_BUFFER, VBO[meshIndex]);
//     glBufferSubData(GL_ARRAY_BUFFER, 0, vertexData.size() * sizeof(float),
//                    vertexData.data());
// }
std::vector<Eigen::Vector3f> Visualization::computeDeformedVertices(
    const std::vector<Eigen::Vector3f>& restPose,
    const std::vector<std::vector<float>>& weights,
    const std::vector<Eigen::Matrix4f>& transforms) {
    
    std::vector<Eigen::Vector3f> deformedVertices(restPose.size());
    
    for(size_t i = 0; i < restPose.size(); i++) {
        Eigen::Vector4f pos(restPose[i].x(), restPose[i].y(), restPose[i].z(), 1.0f);
        Eigen::Vector4f newPos = Eigen::Vector4f::Zero();
        
        for(size_t j = 0; j < weights[i].size(); j++) {
            newPos += weights[i][j] * (transforms[j] * pos);
        }
        
        deformedVertices[i] = Eigen::Vector3f(newPos.x(), newPos.y(), newPos.z());
    }
    
    return deformedVertices;
}
void Visualization::updateBonePositions(
    const std::vector<std::vector<float>>& skinningWeights,
    const std::vector<Eigen::Vector3f>& vertices) {
    
    // First find mesh bounds
    Eigen::Vector3f minBound = vertices[0];
    Eigen::Vector3f maxBound = vertices[0];
    for(const auto& v : vertices) {
        minBound = minBound.cwiseMin(v);
        maxBound = maxBound.cwiseMax(v);
    }
    
    std::vector<float> bonePositions;
    bonePositions.reserve(skinningWeights.size() * 3);
    
    // For each bone
    for(size_t j = 0; j < skinningWeights.size(); j++) {
        Eigen::Vector3f bonePos = Eigen::Vector3f::Zero();
        float totalWeight = 0.0f;
        float maxWeight = 0.0f;
        size_t maxWeightIndex = 0;
        
        // Find vertex with maximum weight for this bone
        for(size_t i = 0; i < vertices.size(); i++) {
            float weight = skinningWeights[j][i];
            if(weight > maxWeight) {
                maxWeight = weight;
                maxWeightIndex = i;
            }
            if(weight > 0.0f) {
                bonePos += weight * vertices[i];
                totalWeight += weight;
            }
        }
        
        // Use position of vertex with maximum weight if total weight is too small
        if(totalWeight > 0.0f) {
            bonePos /= totalWeight;
        } else {
            bonePos = vertices[maxWeightIndex];
        }
        
        // Clamp position within mesh bounds
        bonePos = bonePos.cwiseMax(minBound).cwiseMin(maxBound);
        
        bonePositions.push_back(bonePos.x());
        bonePositions.push_back(bonePos.y());
        bonePositions.push_back(bonePos.z());
    }
    
    normalizeVertices(bonePositions);
    
    glBindBuffer(GL_ARRAY_BUFFER, boneVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, bonePositions.size() * sizeof(float), 
                   bonePositions.data());
}

void Visualization::render(const MeshLoader& loader, std::vector<std::vector<float>> skinningWeights,
    std::vector<std::vector<Eigen::Matrix4f>> boneTransforms) {
    static int currentPose = 0;
    static double lastTime = glfwGetTime();
    static float angle = 0.0f;

    if (!buffersInitialized) {
        setupBuffers(loader);
    }

    angle += 0.01f;
    
    double currentTime = glfwGetTime();
    if (currentTime - lastTime > 2.0) {
        currentPose = (currentPose + 1) % boneTransforms[0].size();
        lastTime = currentTime;
        
        const auto& restVertices = loader.getRestVertices();
        const auto& actualDeformed = loader.getDeformedVertices()[currentPose];
        std::vector<Eigen::Vector3f> calculatedDeformed(restVertices.size());
        
        // Calculate deformed vertices
        for (size_t i = 0; i < restVertices.size(); i++) {
            Eigen::Vector4f restPos(restVertices[i].x(), restVertices[i].y(), restVertices[i].z(), 1.0f);
            Eigen::Vector4f deformedPos = Eigen::Vector4f::Zero();

            for (size_t j = 0; j < skinningWeights.size(); j++) {
                deformedPos += skinningWeights[j][i] * (boneTransforms[j][currentPose] * restPos);
            }
            
            calculatedDeformed[i] = Eigen::Vector3f(deformedPos.x(), deformedPos.y(), deformedPos.z());
        }

        // Calculate error metrics
        float totalError = 0.0f;
        float maxError = 0.0f;
        
        for(size_t i = 0; i < restVertices.size(); i++) {
            float error = (calculatedDeformed[i] - actualDeformed[i]).norm();
            totalError += error;
            maxError = std::max(maxError, error);
        }
        
        float avgError = totalError / restVertices.size();
        
        // Print error metrics
        std::cout << "\rFrame " << currentPose 
                 << " - Average Error: " << std::fixed << std::setprecision(6) << avgError
                 << " Max Error: " << std::fixed << std::setprecision(6) << maxError 
                 << std::flush;

        // Update bone positions
        updateBonePositions(skinningWeights, calculatedDeformed);
    
        // Update both meshes
        updateMesh(calculatedDeformed, 0, loader);  // Calculated mesh on left
        updateMesh(actualDeformed, 1, loader);      // Actual mesh on right
    }

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shaderProgram);

    // Set up base view and projection matrices
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
    view.block<3,1>(0,3) = Eigen::Vector3f(0.0f, 0.0f, -3.0f);

    float aspect = 1200.0f / 600.0f;
    Eigen::Matrix4f projection = Eigen::Matrix4f::Zero();
    float fov = 45.0f * M_PI / 180.0f;
    float tanHalfFov = std::tan(fov / 2.0f);
    projection(0,0) = 1.0f / (aspect * tanHalfFov);
    projection(1,1) = 1.0f / tanHalfFov;
    projection(2,2) = -(100.0f + 0.1f) / (100.0f - 0.1f);
    projection(2,3) = -(2.0f * 100.0f * 0.1f) / (100.0f - 0.1f);
    projection(3,2) = -1.0f;

    // Set up model matrix with rotation
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    model.block<3,3>(0,0) = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitY()).matrix();

    // Common uniform locations
    GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
    GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
    GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
    GLint viewPosLoc = glGetUniformLocation(shaderProgram, "viewPos");
    GLint lightPosLoc = glGetUniformLocation(shaderProgram, "lightPos");
    GLint meshColorLoc = glGetUniformLocation(shaderProgram, "meshColor");

    // Common lighting parameters
    Eigen::Vector3f viewPos(0.0f, 0.0f, 3.0f);
    Eigen::Vector3f lightPos(2.0f, 2.0f, 2.0f);

    glUniform3fv(viewPosLoc, 1, viewPos.data());
    glUniform3fv(lightPosLoc, 1, lightPos.data());

    // Draw calculated mesh (left side)
    glViewport(0, 0, 600, 600);  // Left half of the window
    
    Eigen::Matrix4f leftModel = model;
    leftModel.block<3,1>(0,3) = Eigen::Vector3f(-0.5f, 0.0f, 0.0f);  // Shift left
    
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, leftModel.data());
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view.data());
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projection.data());
    
    Eigen::Vector3f calculatedColor(0.7f, 0.2f, 0.2f);  // Red for calculated
    glUniform3fv(meshColorLoc, 1, calculatedColor.data());
    
    glBindVertexArray(VAO[0]);
    glDrawElements(GL_TRIANGLES, loader.getFaces().size() * 3, GL_UNSIGNED_INT, 0);

    // Draw bones only for left viewport (calculated mesh)
    glDisable(GL_DEPTH_TEST);  // Disable depth test for bones
    glPointSize(10.0f);  // Make points larger
    Eigen::Vector3f boneColor(1.0f, 1.0f, 0.0f);  // Yellow for bones
    glUniform3fv(meshColorLoc, 1, boneColor.data());
    
    glBindVertexArray(boneVAO);
    glDrawArrays(GL_POINTS, 0, skinningWeights.size());

    // Draw actual mesh (right side)
    glViewport(600, 0, 600, 600);  // Right half of the window
    
    Eigen::Matrix4f rightModel = model;
    rightModel.block<3,1>(0,3) = Eigen::Vector3f(0.5f, 0.0f, 0.0f);  // Shift right
    
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, rightModel.data());
    
    Eigen::Vector3f actualColor(0.2f, 0.7f, 0.2f);  // Green for actual
    glUniform3fv(meshColorLoc, 1, actualColor.data());
    
    glBindVertexArray(VAO[1]);
    glDrawElements(GL_TRIANGLES, loader.getFaces().size() * 3, GL_UNSIGNED_INT, 0);

    glfwSwapBuffers(window);
    glfwPollEvents();
}

Visualization::~Visualization() {
    if(buffersInitialized) {
        glDeleteVertexArrays(2, VAO);
        glDeleteBuffers(2, VBO);
        glDeleteBuffers(2, EBO);
        glDeleteVertexArrays(1, &boneVAO);  // Add this
        glDeleteBuffers(1, &boneVBO);       // Add this
    }
    glDeleteProgram(shaderProgram);
    glfwTerminate();
}