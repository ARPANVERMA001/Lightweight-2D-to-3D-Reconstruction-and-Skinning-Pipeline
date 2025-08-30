.SUFFIXES:

# Suppress CUDA warnings and allow relaxed constexpr
NVCC_FLAGS = --expt-relaxed-constexpr -diag-suppress 940
CXX_FLAGS = -Wno-cpp -Wno-deprecated-declarations -Wno-return-type

nvcc = /usr/local/cuda-12.6/bin/nvcc
cudalib = /usr/local/cuda-12.6/lib64

# Path to TensorFlow include and library
tensorflow = /home/chandan/miniconda3/envs/pixel2mesh_env/lib/python2.7/site-packages/tensorflow/include
TENSORFLOW_LIB = $(shell python -c "import tensorflow as tf; print(tf.sysconfig.get_lib())")

all: tf_approxmatch_so.so tf_approxmatch_g.cu.o tf_nndistance_so.so tf_nndistance_g.cu.o


tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++11 $(CXX_FLAGS) tf_approxmatch.cpp tf_approxmatch_g.cu.o \
		-I $(tensorflow) -L $(cudalib) -lcudart -L $(TENSORFLOW_LIB) -ltensorflow_framework \
		-O2 -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -o tf_approxmatch_so.so


tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 $(NVCC_FLAGS) -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2


tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 $(CXX_FLAGS) tf_nndistance.cpp tf_nndistance_g.cu.o \
		-I $(tensorflow) -L $(cudalib) -lcudart -L $(TENSORFLOW_LIB) -ltensorflow_framework \
		-O2 -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -o tf_nndistance_so.so


tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 $(NVCC_FLAGS) -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2


clean:
	rm tf_approxmatch_so.so
	rm tf_nndistance_so.so
	rm  *.cu.o