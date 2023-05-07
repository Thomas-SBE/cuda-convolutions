CXX=g++
NVC=nvcc
CXXFLAGS=-O3 -march=native
LDLIBS=-lm -lIL

all: blur sobel edge laplacian
cuda: cuda_edge cuda_sobel cuda_blur cuda_laplacian

blur: seq/blur.cpp
	$(CXX) $(CXXFLAGS) -o build/$@ $< $(LDLIBS)

sobel: seq/sobel_edge.cpp
	$(CXX) $(CXXFLAGS) -o build/$@ $< $(LDLIBS)

edge: seq/edge.cpp
	$(CXX) $(CXXFLAGS) -o build/$@ $< $(LDLIBS)

laplacian: seq/laplacian.cpp
	$(CXX) $(CXXFLAGS) -o build/$@ $< $(LDLIBS)

cuda_edge: cuda/edge.cu
	$(NVC) -o build/$@ $< $(LDLIBS)

cuda_sobel: cuda/sobel_edge.cu
	$(NVC) -o build/$@ $< $(LDLIBS)

cuda_blur: cuda/blur.cu
	$(NVC) -o build/$@ $< $(LDLIBS)

cuda_laplacian: cuda/laplacian.cu
	$(NVC) -o build/$@ $< $(LDLIBS)

unoptimized: unoptimized/edge.cu
	$(NVC) -o build/$@ $< $(LDLIBS)

l_unoptimized: unoptimized/laplacian.cu
	$(NVC) -o build/$@ $< $(LDLIBS)

b_unoptimized: unoptimized/blur.cu
	$(NVC) -o build/$@ $< $(LDLIBS)

shared: optimized/edge_sm.cu
	$(NVC) -o build/$@ $< $(LDLIBS)

l_shared: optimized/laplacian_sm.cu
	$(NVC) -o build/$@ $< $(LDLIBS)

b_shared: optimized/blur_sm.cu
	$(NVC) -lineinfo -o build/$@ $< $(LDLIBS)

details: details.cu
	$(NVC) -o build/$@ $< $(LDLIBS)


.PHONY: clean

clean_cuda:
	rm cuda_edge cuda_sobel cuda_blur cuda_laplacian
clean:
	rm blur sobel laplacian edge