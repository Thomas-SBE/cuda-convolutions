CXX=g++
NVC=nvcc
CXXFLAGS=-O3 -march=native
LDLIBS=-lm -lIL

all: blur sobel edge laplacian
cuda: cuda_edge

blur: blur.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

sobel: sobel_edge.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

edge: edge.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

laplacian: laplacian.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

cuda_edge: edge.cu
	$(NVC) -o $@ $< $(LDLIBS)

.PHONY: clean

clean_cuda:
	rm cuda_edge
clean:
	rm blur sobel laplacian edge