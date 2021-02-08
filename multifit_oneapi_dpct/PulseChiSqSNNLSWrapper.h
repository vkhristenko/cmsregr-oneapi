#ifndef PULSECHISQSNNLSWRAPPER
#define PULSECHISQSNNLSWRAPPER

// #define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
// #define EIGEN_DONT_VECTORIZE 

#include <vector>

#include "DeviceData.h"

std::vector<Output> doFitWrapper(std::vector<DoFitArgs> const&);

#endif
