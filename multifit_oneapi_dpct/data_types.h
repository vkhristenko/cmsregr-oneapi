#ifndef NNLS_DATA_TYPES
#define NNLS_DATA_TYPES

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <Eigen/Dense>

constexpr unsigned long MATRIX_SIZE = 10;
constexpr unsigned long VECTOR_SIZE = 10;

typedef Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE, Eigen::ColMajor>
    FixedMatrix;
typedef Eigen::Matrix<double, VECTOR_SIZE, 1> FixedVector;

typedef Eigen::PermutationMatrix<VECTOR_SIZE> FixedPermutation;

typedef Eigen::Matrix<double, -1, -1, 0, 10, 10> FixedDynamicMatrix;
typedef Eigen::Matrix<double, -1, 1, 0, 10, 1> FixedDynamicVector;

#define USE_LDLT 0
#define USE_LLT 1
#define USE_HOUSEHOLDER 2

#ifndef DECOMPOSITION
#define DECOMPOSITION USE_LLT
#endif

#include <iostream>

template <typename M, typename V>
#ifdef DPCT_COMPATIBILITY_TEMP

#endif
    Eigen::Matrix<typename M::Scalar,
                  Eigen::Dynamic,
                  Eigen::Dynamic,
                  M::Options,
                  M::RowsAtCompileTime,
                  M::ColsAtCompileTime> inline sub_matrix(const M& full,
                                                          const V& index,
                                                          unsigned int size) {
  using matrix_t =
      Eigen::Matrix<typename M::Scalar, Eigen::Dynamic, Eigen::Dynamic,
                    M::Options, M::RowsAtCompileTime, M::ColsAtCompileTime>;
  matrix_t _matrix(size, size);

#pragma unroll
  for (auto i = 0, idx = 0; i < VECTOR_SIZE; i++) {
    if (!index[i]) {
//#pragma vectorise
      for (auto j = 0, idy = 0; j < VECTOR_SIZE; j++) {
        if (!index[j])
          _matrix(idx, idy++) = full(i, j);
      }
      ++idx;
    }
  }
  return _matrix;
}

template <typename M, typename V>
#ifdef DPCT_COMPATIBILITY_TEMP

#endif
    Eigen::Matrix<typename M::Scalar,
                  Eigen::Dynamic,
                  1,
                  M::Options,
                  M::RowsAtCompileTime,
                  1> inline sub_vector(const M& full,
                                       const V& index,
                                       unsigned int size) {
  using matrix_t = Eigen::Matrix<typename M::Scalar, Eigen::Dynamic, 1,
                                 M::Options, M::RowsAtCompileTime, 1>;
  matrix_t _vector(size);
#pragma unroll
  for (auto i = 0, idx = 0; i < VECTOR_SIZE; i++) {
    if (!index[i])
      _vector[idx++] = full[i];
  }
  return _vector;
}

typedef struct NNLS_args {
  FixedMatrix const A;
  FixedVector const b;
#ifdef DPCT_COMPATIBILITY_TEMP

#endif
  NNLS_args(FixedMatrix const A, FixedVector const b)
      : A(A), b(b){};
} NNLS_args;
// #endif

#endif
