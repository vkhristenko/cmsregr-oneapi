.PHONY: all clean

CXX=g++
CXXFLAGS=--std=c++17 -O2

HEADERS=$(shell ls *.h)

all: multifit

multifit: multifit.o PulseChiSqSNNLS.o inplace_fnnls.o
	$(CXX) $^ $(LDFLAGS) $(ROOT_LDFLAGS) $(ROOT_LIBS) -o $@

%.o: %.cpp $(HEADERS) 
	$(CXX) $(CXXFLAGS) $(ROOT_CXXFLAGS) -I$(EIGEN_HOME) -c $< -o $@

clean:
	rm *.o multifit
