//; $ROW = param_define("ROW", 4);
//; $COL = param_define("COL", 3);
//; $BIT_R = param_define("BIT_R", 3);
//; $BIT_C = param_define("BIT_C", 3);
//This program computes QR decomposition of A
//using Givens Rotations with parallel rotation generation.

#include "qr.h"
#include <iostream>

using namespace std;

MATRIX_T qrf_mag(MATRIX_T a, MATRIX_T b)
{
#pragma HLS inline
	MATRIX_T aa = a * a;
	MATRIX_T bb = b * b;
	MATRIX_T mag = x_sqrt(aa + bb);
	return mag;
}

/*void qrf_givens(MATRIX_T x, MATRIX_T y, MATRIX_T &c, MATRIX_T &s, MATRIX_T &r)
{
	r = qrf_mag(x, y);

	c = x / r;
	s = y / r;
}*/

//can be used for both left mm and right mm
//for left mm:
// [a b]|c -s|
//		|s  c|
//for right mm:
// | c s||a|
// |-s c||b|
void qrf_mm(MATRIX_T c, MATRIX_T s, MATRIX_T &op1, MATRIX_T &op2)
{
#pragma HLS inline
	MATRIX_T a = op2 * s + op1 * c;
	MATRIX_T b = op2 * c - op1 * s;

	op1 = a;
	op2 = b;
}

void feeder(MATRIX_T A[ROWS][COLS],
			stream<MATRIX_T> &feedin)
{
	for(uint_i i = 0; i < COLS; i++)
	{
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			feedin.write(A[j][i]);
		}
	}
}

void PE_head(stream<MATRIX_T> &in_A,
			stream<MATRIX_T> &out_A,
			stream<MATRIX_T> &pass_c,
			stream<MATRIX_T> &pass_s,
			stream<MATRIX_T> &out_R)
{
	MATRIX_T A[ROWS];
	MATRIX_T A_temp;

	//for operand A, read and pass
	for(uint_i i = 0; i < COLS; i++)
	{
		for(uint_i j = 0 ; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			if(i == 0)
			{
				A[j] = in_A.read();
			}
			else
			{
				A_temp = in_A.read();
				out_A.write(A_temp);
			}
		}
	}

	//angle generation: step1
	for(uint_i i = ROWS - 1; i >= UNROLL_FACTOR; i--)
	{
#pragma HLS unroll factor=2
		if(hls::abs(A[i]) < 1e-6) //rounding error
		{
			MATRIX_T c = 1;
			MATRIX_T s = 0;
			pass_c.write(c);
			pass_s.write(s);
		}
		else
		{
			uint_i k = i - UNROLL_FACTOR;
			MATRIX_T mag = qrf_mag(A[i], A[k]);
			MATRIX_T c = A[k] / mag;
			MATRIX_T s = A[i] / mag;
			A[k] = mag;
			A[i] = 0;
			pass_c.write(c);
			pass_s.write(s);
		}
	}
	//step2:
	for(uint_i i = UNROLL_FACTOR - 1; i > 0; i--)
	{
		if(hls::abs(A[i]) < 1e-6) //rounding error
		{
			MATRIX_T c = 1;
			MATRIX_T s = 0;
			pass_c.write(c);
			pass_s.write(s);
			}
		else
		{
			MATRIX_T mag = qrf_mag(A[i], A[i - 1]);
			MATRIX_T c = A[i - 1] / mag;
			MATRIX_T s = A[i] / mag;
			A[i - 1] = mag;
			A[i] = 0;
			pass_c.write(c);
			pass_s.write(s);
		}
	}
	//output A
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		out_R.write(A[i]);
	}
}

//id starts from 1, ends at (COLS - 2)
void PE(uint_i id,
		stream<MATRIX_T> &in_A,
		stream<MATRIX_T> &out_A,
		stream<MATRIX_T> &in_c,
		stream<MATRIX_T> &in_s,
		stream<MATRIX_T> &out_c,
		stream<MATRIX_T> &out_s,
		stream<MATRIX_T> &out_R)
{
	MATRIX_T A[ROWS];

	//for operand A, read and pass
	for(uint_i i = id; i < COLS; i++)
	{
		for(uint_i j = 0 ; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			if(i == id)
			{
				A[j] = in_A.read();
			}
			else
			{
				out_A.write(in_A.read());
			}
		}
	}
	//rotation of rows: step1
	for(uint_i j = ROWS - 1; j >= UNROLL_FACTOR; j--)
	{
#pragma HLS unroll factor=2
		uint_i k = j - UNROLL_FACTOR;
		MATRIX_T c = in_c.read();
		MATRIX_T s = in_s.read();
		qrf_mm(c, s, A[k], A[j]);
		out_c.write(c);
		out_s.write(s);
	}
	for(uint_i j = UNROLL_FACTOR - 1; j > 0; j--)
	{
		MATRIX_T c = in_c.read();
		MATRIX_T s = in_s.read();
		qrf_mm(c, s, A[j - 1], A[j]);
		out_c.write(c);
		out_s.write(s);
	}
	//rotation of rows: step2
	for(uint_i i = 1; i < id; i++)
	{
		for(uint_i j = ROWS - 1; j > i; j--)
		{
#pragma HLS unroll factor=2
			MATRIX_T c = in_c.read();
			MATRIX_T s = in_s.read();
			qrf_mm(c, s, A[j - UNROLL_FACTOR], A[j]);
			out_c.write(c);
			out_s.write(s);
		}
	}

	//angle generation
	for(uint_i i = ROWS - 1; i > id; i--)
	{
#pragma HLS unroll factor=2
		if(hls::abs(A[i]) < 1e-6)
		{
			MATRIX_T c = 1;
			MATRIX_T s = 0;
			out_c.write(c);
			out_s.write(s);
		}
		else
		{
			MATRIX_T mag = qrf_mag(A[i], A[i - UNROLL_FACTOR]);
			MATRIX_T c = A[i - UNROLL_FACTOR] / mag;
			MATRIX_T s = A[i] / mag;
			A[i - UNROLL_FACTOR] = mag;
			A[i] = 0;
			out_c.write(c);
			out_s.write(s);
		}
	}
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		out_R.write(A[i]);
	}
}

void PE_tail(
		stream<MATRIX_T> &in_A,
		stream<MATRIX_T> &in_c,
		stream<MATRIX_T> &in_s,
		stream<MATRIX_T> &out_c,
		stream<MATRIX_T> &out_s,
		stream<MATRIX_T> &out_R)
{
	MATRIX_T A[ROWS];

	//for operand A
	for(uint_i j = 0 ; j < ROWS; j++)
	{
#pragma HLS pipeline II=1
		A[j] = in_A.read();
	}

	//rotation of rows: step1
	for(uint_i j = ROWS - 1; j >= UNROLL_FACTOR; j--)
	{
#pragma HLS unroll factor=2
		uint_i k = j - UNROLL_FACTOR;
		MATRIX_T c = in_c.read();
		MATRIX_T s = in_s.read();
		qrf_mm(c, s, A[k], A[j]);
		out_c.write(c);
		out_s.write(s);
	}
	for(uint_i j = UNROLL_FACTOR - 1; j > 0; j--)
	{
		MATRIX_T c = in_c.read();
		MATRIX_T s = in_s.read();
		qrf_mm(c, s, A[j - 1], A[j]);
		out_c.write(c);
		out_s.write(s);
	}
	//rotation of rows: step2
	for(uint_i i = 1; i < COLS - 1; i++)
	{
		for(uint_i j = ROWS - 1; j > i; j--)
		{
#pragma HLS unroll factor=2
			MATRIX_T c = in_c.read();
			MATRIX_T s = in_s.read();
			qrf_mm(c, s, A[j - UNROLL_FACTOR], A[j]);
			out_c.write(c);
			out_s.write(s);
		}
	}

	//angle generation
	for(uint_i i = ROWS - 1; i >= COLS; i--)
	{
#pragma HLS unroll factor=2
		if(hls::abs(A[i]) < 1e-6)
		{
			MATRIX_T c = 1;
			MATRIX_T s = 0;
			out_c.write(c);
			out_s.write(s);
		}
		else
		{
			MATRIX_T mag = qrf_mag(A[i], A[i - UNROLL_FACTOR]);
			MATRIX_T c = A[i - UNROLL_FACTOR] / mag;
			MATRIX_T s = A[i] / mag;
			A[i - UNROLL_FACTOR] = mag;
			A[i] = 0;
			out_c.write(c);
			out_s.write(s);
		}
	}

	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		out_R.write(A[i]);
	}
}

void collector(stream<MATRIX_T> &in_c,
				stream<MATRIX_T> &in_s,
//; for($i = 0; $i < $COL; $i++)
//; {
				stream<MATRIX_T> &in_R`$i`,
//; }
				MATRIX_T Q[ROWS][ROWS],
				MATRIX_T R[ROWS][COLS])
{
	//initialize for Q
	MATRIX_T Q_i[ROWS][ROWS];

	for(uint_i i = 0; i < ROWS; i++)
	{
		for (uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			if(i == j)
			{
				Q_i[i][j] = 1;
			}
			else
			{
				Q_i[i][j] = 0;
			}
		}
	}

//; for($i = 0; $i < $COL; $i++)
//; {
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		R[i][`$i`] = in_R`$i`.read();
	}
//; }

	//rotation of rows: step1
		for(uint_i j = ROWS - 1; j >= UNROLL_FACTOR; j--)
		{
	#pragma HLS unroll factor=2
			uint_i k = j - UNROLL_FACTOR;
			MATRIX_T c = in_c.read();
			MATRIX_T s = in_s.read();
			for(uint_i k = 0; k < ROWS; k++)
			{
				qrf_mm(c, s, Q_i[k][j - UNROLL_FACTOR], Q_i[k][j]);
			}
		}
		for(uint_i j = UNROLL_FACTOR - 1; j > 0; j--)
		{
			MATRIX_T c = in_c.read();
			MATRIX_T s = in_s.read();
			for(uint_i k = 0; k < ROWS; k++)
			{
				qrf_mm(c, s, Q_i[k][j - 1], Q_i[k][j]);
			}
		}
		//rotation of rows: step2
		for(uint_i i = 1; i < COLS; i++)
		{
			for(uint_i j = ROWS - 1; j > i; j--)
			{
	#pragma HLS unroll factor=2
				MATRIX_T c = in_c.read();
				MATRIX_T s = in_s.read();
				for(uint_i k = 0; k < ROWS; k++)
				{
					qrf_mm(c, s, Q_i[k][j - UNROLL_FACTOR], Q_i[k][j]);
				}
			}
		}

	for(uint_i i = 0; i < ROWS; i++)
	{
		for (uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			Q[i][j] = Q_i[i][j];
		}
	}
}

int top(MATRIX_T A[ROWS][COLS],
	MATRIX_T Q[ROWS][ROWS],
	MATRIX_T R[ROWS][COLS])
{
	#pragma HLS dataflow

	stream<MATRIX_T> feedin;
#pragma HLS stream variable=feedin depth=10

//; $COL_1 = $COL - 1;
//; for($i = 0; $i < $COL_1; $i++)
//; {
	stream<MATRIX_T> out_A`$i`;
#pragma HLs stream variable=out_A`$i` depth=10
//; }

//; for($i = 0; $i < $COL; $i++)
//; {
	stream<MATRIX_T> pass_c`$i`;
#pragma HLS stream variable=pass_c`$i` depth=10
//; }

//; for($i = 0; $i < $COL; $i++)
//; {
	stream<MATRIX_T> pass_s`$i`;
#pragma HLS stream variable=pass_s`$i` depth=10
//; }

//; for($i = 0; $i < $COL; $i++)
//; {
	stream<MATRIX_T> out_R`$i`;
#pragma HLS stream variable=out_R`$i` depth=10
//; }

	feeder(A, feedin);
	PE_head(feedin, out_A0, pass_c0, pass_s0, out_R0);
//; $index = 1;
//; for($index = 1; $index < $COL - 1; $index++)
//; {
//; $index_1 = $index - 1;
	PE(`$index`, out_A`$index_1`, out_A`$index`, pass_c`$index_1`, pass_s`$index_1`, pass_c`$index`, pass_s`$index`, out_R`$index`);
//; }
//; $index = $index - 1;
//; $index_p1 = $index + 1;
	PE_tail(out_A`$index`, pass_c`$index`, pass_s`$index`, pass_c`$index_p1`, pass_s`$index_p1`, out_R`$index_p1`);
	collector(pass_c`$index_p1`, pass_s`$index_p1`,
//; for($i = 0; $i < $COL; $i++)
//; {
		 out_R`$i`,
//; }
		 Q, R);
	return 0;
}
