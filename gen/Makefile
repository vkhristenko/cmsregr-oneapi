.PHONY: all clean

CXX=g++
CXXFLAGS=--std=c++17

all: gen_data

gen_data: gen_data.o
	$(CXX) $^ $(LDFLAGS) $(ROOT_LDFLAGS) $(ROOT_LIBS) -o $@

%.o : %.cpp Pulse.h
	$(CXX) $(CXXFLAGS) $(ROOT_CXXFLAGS) -c $< -o $@

clean:
	rm *.o gen_data
