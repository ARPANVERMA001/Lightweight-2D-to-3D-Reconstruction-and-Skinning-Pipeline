# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall

# Include directories
INCLUDES = -I/usr/include/eigen3 -I/usr/include/glm

# Libraries
LIBS = -lglfw -L/usr/lib64 -lGLEW -lGL -lX11 -lGLU -lGL -ltinyobjloader

# Source directory
SRCDIR = src

# Source files
SOURCES = main.cpp \
          $(SRCDIR)/MeshLoader.cpp \
          $(SRCDIR)/Visualization.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Target executable
MAIN_TARGET = main_app

# Default target
all: $(MAIN_TARGET)

# Linking rule
$(MAIN_TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(MAIN_TARGET) $(LIBS)

# Compilation rule
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Run the executable
run: $(MAIN_TARGET)
	./$(MAIN_TARGET)

# Clean rule
clean:
	rm -f $(OBJECTS) $(MAIN_TARGET)

# Phony targets
.PHONY: all clean run