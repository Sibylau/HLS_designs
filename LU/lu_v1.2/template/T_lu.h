//; $SIZE = param_define("SIZE", 6);
//; $BIT = param_define("BIT", 3);
/*******LU decomposition********
* header file, Jie Liu, 09/02/2018 */

#ifndef LU_H
#define LU_H
#include "hls_stream.h"
#include "ap_int.h"
#include "hls_linear_algebra.h"
#include "hls_math.h"
#define matrix_size `$SIZE`
#define iterator_bit `$BIT`
#define fifo_length (matrix_size + 1) * matrix_size / 2

using namespace hls;

typedef float matrix_t;
typedef ap_uint<iterator_bit> uint_i;

int top(matrix_t A[matrix_size][matrix_size],
	    matrix_t L[matrix_size][matrix_size],
	    matrix_t U[matrix_size][matrix_size],
	    uint_i P[matrix_size]);

#endif
