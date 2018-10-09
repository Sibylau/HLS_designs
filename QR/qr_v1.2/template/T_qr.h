//; $ROW = param_define("ROW", 4);
//; $COL = param_define("COL", 3);
//; $BIT_R = param_define("BIT_R", 3);
//; $BIT_C = param_define("BIT_C", 3);
/*******QR decomposition********
* header file, Jie Liu, 09/08/2018 */

#ifndef qr_GR_H
#define qr_GR_H
#include "hls_stream.h"
#include "ap_int.h"
#include "hls_linear_algebra.h"
#include "hls_math.h"
#define ROWS `$ROW`
#define COLS `$COL`
#define ib_R `$BIT_R`
#define ib_C `$BIT_C`
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

