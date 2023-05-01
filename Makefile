CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS=-lm -lIL

all: blur sobel edge laplacian

blur: blur.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

sobel: sobel_edge.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

edge: edge.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

laplacian: laplacian.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm blur