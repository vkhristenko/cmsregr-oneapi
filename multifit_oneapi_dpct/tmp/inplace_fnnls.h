#ifndef ONE_PASS_FNNLS_H
#define ONE_PASS_FNNLS_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "data_types.h"

SYCL_EXTERNAL
void
inplace_fnnls(const FixedMatrix& A,
                  const FixedVector& b,
                  FixedVector& x,
                  const double eps = 1e-11,
                  const unsigned int max_iterations = 1000);

#endif
