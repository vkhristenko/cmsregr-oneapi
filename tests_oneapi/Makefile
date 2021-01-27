.PHONY: all clean

SOURCES=$(shell ls *.cpp)
TARGETS=$(SOURCES:.cpp=)
GCC_HOME=$(shell which g++)/../../

#$(info SOURCES=$(SOURCES))
#$(info TARGETS=$(TARGETS))

CXX = dpcpp
CXXFLAGS = -O2 -g -std=c++17 --gcc-toolchain=$(GCC_HOME)

all: $(TARGETS)

$(TARGETS): %: %.o
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm $(TARGETS) *.o
