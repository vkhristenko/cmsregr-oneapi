.PHONY: all clean

GCC_HOME=$(shell which g++)/../../
CXX=dpcpp
CXXFLAGS=--std=c++17 -O2 --gcc-toolchain=$(GCC_HOME) -DEIGEN_DONT_VECTORIZE -DEIGEN_USE_SYCL -DEIGEN_NO_MALLOC -DEIGEN_NO_CPUID -Wall -pedantic #-Dasm=error -D__asm__=error

HEADERS=$(shell ls *.h)
SRCS=$(shell ls *.cpp)
OBJS=$(SRCS:.cpp=.o)

all: multifit

multifit: $(OBJS)
	$(CXX) $^ -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice $(LDFLAGS) $(ROOT_LDFLAGS) $(ROOT_LIBS) -o $@

%.o: %.cpp $(HEADERS) 
	$(CXX) $(CXXFLAGS) $(ROOT_CXXFLAGS) -I$(EIGEN_HOME) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -c $< -o $@

clean:
	rm *.o multifit
