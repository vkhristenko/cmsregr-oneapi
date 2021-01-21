.PHONY: all clean

CXX=g++
CXXFLAGS=--std=c++17

all: gen_data

gen_data: gen_data.o
	@echo "Building executable: gen_data"
	@echo "----------------------------"
	$(CXX) $^ $(LDFLAGS) $(ROOT_LDFLAGS) $(ROOT_LIBS) -o $@

%.o : %.cpp Pulse.h
	@echo "Compiling source file: $<"
	@echo "----------------------------"
	$(CXX) $(CXXFLAGS) $(ROOT_CXXFLAGS) -c $< -o $@

clean:
	rm gen_data gen_data.o
