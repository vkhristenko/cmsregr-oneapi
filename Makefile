.PHONY: all clean gen multifit_cpp

export ROOT_LIBS=$(shell root-config --libs)
export ROOT_CXXFLAGS=$(shell root-config --cflags)
export ROOT_LDFLAGS=$(shell root-config --ldflags)

export EIGEN_HOME=/home/eigen

all: gen multifit_cpp

multifit_cpp:
	cd multifit_cpp && $(MAKE)

gen:
	cd $@ && $(MAKE)

clean: clean_gen clean_multifit_cpp

clean_gen:
	cd gen/ && $(MAKE) clean

clean_multifit_cpp:
	cd multifit_cpp && $(MAKE) clean
