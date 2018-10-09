/*******Cholesky decomposition of LL^T form********
* header file, Jie Liu, 7/26/2018 */
#ifndef CHOLESKY_H
#define CHOLESKY_H
#include "hls_stream.h"
#include "ap_int.h"
#include "hls_linear_algebra.h"
#include "hls_math.h"
#define matrix_size 4
#define iterator_bit 3
#define max_length (matrix_size - 1) * matrix_size / 2
using namespace hls;

typedef float matrix_t;
int top(matrix_t A[matrix_size][matrix_size],
	    matrix_t L[matrix_size][matrix_size]);

#endif
