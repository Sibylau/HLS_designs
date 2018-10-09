/*******QR decomposition********
* header file, Jie Liu, 09/08/2018 */

#ifndef qr_GR_H
#define qr_GR_H
#include "hls_stream.h"
#include "ap_int.h"
#include "hls_linear_algebra.h"
#include "hls_math.h"
#define ROWS 4
#define COLS 4
#define ib_R 3
#define ib_C 3
#define fifo_length (ROWS + 1) * ROWS / 2
#define UNROLL_FACTOR 2
#include <iostream>

using namespace hls;

typedef float MATRIX_T;
typedef ap_uint<ib_R> uint_i;

int top(MATRIX_T A[ROWS][COLS],
	    MATRIX_T Q[ROWS][ROWS],
	    MATRIX_T R[ROWS][COLS]);
MATRIX_T qrf_mag(MATRIX_T a, MATRIX_T b);
void qrf_mm(MATRIX_T c, MATRIX_T s, MATRIX_T &op1, MATRIX_T &op2);

#endif

