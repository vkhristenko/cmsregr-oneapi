.PHONY: all clean

CXX=g++
CXXFLAGS=--std=c++17

all: test

test: test.o
	$(CXX) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@echo call make clean
	rm test
