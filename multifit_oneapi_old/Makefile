.PHONY: all clean

GCC_HOME=$(shell which g++)/../../
CXX=dpcpp
CXXFLAGS=--std=c++17 --gcc-toolchain=$(GCC_HOME)

HEADERS=$(shell ls *.h)

all: multifit

multifit: multifit.o PulseChiSqSNNLS.o inplace_fnnls.o
	$(CXX) $^ $(LDFLAGS) $(ROOT_LDFLAGS) $(ROOT_LIBS) -o $@

%.o: %.cpp $(HEADERS) 
	$(CXX) $(CXXFLAGS) $(ROOT_CXXFLAGS) -I$(EIGEN_HOME) -c $< -o $@

clean:
	rm *.o multifit
