.PHONY: all clean gen multifit_cpp tests_oneapi multifit_oneapi

export ROOT_LIBS=$(shell root-config --libs)
export ROOT_CXXFLAGS=$(shell root-config --cflags)
export ROOT_LDFLAGS=$(shell root-config --ldflags)

export EIGEN_HOME=/home/eigen

all: gen multifit_cpp tests_oneapi multifit_oneapi

multifit_cpp:
	cd multifit_cpp && $(MAKE)

multifit_oneapi:
	cd multifit_oneapi && $(MAKE)

tests_oneapi:
	cd tests_oneapi && $(MAKE)

gen:
	cd $@ && $(MAKE)

clean: clean_gen clean_multifit_cpp clean_tests_oneapi clean_multifit_oneapi

clean_gen:
	cd gen/ && $(MAKE) clean

clean_multifit_cpp:
	cd multifit_cpp && $(MAKE) clean

clean_multifit_oneapi:
	cd multifit_oneapi && $(MAKE) clean

clean_tests_oneapi:
	cd tests_oneapi && $(MAKE) clean
