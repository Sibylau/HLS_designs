
//This program computes QR decomposition of A
//using Givens Rotations with parallel rotation generation.
//; $ROW = param_define("ROW", 4);
//; $COL = param_define("COL", 3);
//; $BIT_R = param_define("BIT_R", 3);
//; $BIT_C = param_define("BIT_C", 3);
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
			stream<MATRIX_T> &out_R,
			stream<MATRIX_T> &Q_out)
{
	MATRIX_T A[ROWS];
	MATRIX_T A_temp;
	MATRIX_T Q[ROWS][ROWS];
	MATRIX_T C[2];
	MATRIX_T S[2];

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
	for(uint_i i = 0; i < ROWS; i++)
	{
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			if(i == j)
			{
				Q[i][j] = 1;
			}
			else
			{
				Q[i][j] = 0;
			}
		}
	}
	//angle generation: step1
	bool index = 0;
	for(uint_i i = ROWS - 1; i >= UNROLL_FACTOR; i--)
	{
#pragma HLS pipeline II=1
		if(hls::abs(A[i]) < 1e-6) //rounding error
		{
			C[index] = 1;
			S[index] = 0;
			for(uint_i k = 0; k < ROWS; k++)
			{
				qrf_mm(C[index], S[index], Q[k][i - UNROLL_FACTOR], Q[k][i]);
			}
		}
		else
		{
			uint_i k = i - UNROLL_FACTOR;
			MATRIX_T mag = qrf_mag(A[i], A[k]);
			C[index] = A[k] / mag;
			S[index] = A[i] / mag;
			A[k] = mag;
			A[i] = 0;
			for(uint_i k = 0; k < ROWS; k++)
			{
				qrf_mm(C[index], S[index], Q[k][i - UNROLL_FACTOR], Q[k][i]);
			}
		}
		pass_c.write(C[index]);
		pass_s.write(S[index]);
		index = ~index;
	}
	//step2:
	for(uint_i i = UNROLL_FACTOR - 1; i > 0; i--)
	{
		if(hls::abs(A[i]) < 1e-6) //rounding error
		{
			MATRIX_T c = 1;
			MATRIX_T s = 0;
			for(uint_i k = 0; k < ROWS; k++)
			{
				qrf_mm(c, s, Q[k][i - 1], Q[k][i]);
			}
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
			for(uint_i k = 0; k < ROWS; k++)
			{
				qrf_mm(c, s, Q[k][i - 1], Q[k][i]);
			}
			pass_c.write(c);
			pass_s.write(s);
		}
	}
	for(uint_i i = 0; i < ROWS; i++)
	{
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			Q_out.write(Q[i][j]);
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
		stream<MATRIX_T> &in_R,
		stream<MATRIX_T> &out_R,
		stream<MATRIX_T> &Q_in,
		stream<MATRIX_T> &Q_out)
{
	MATRIX_T A[ROWS];
	MATRIX_T C[2], S[2];
	MATRIX_T Q[ROWS][ROWS];

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
	for(uint_i i = 0; i < ROWS; i++)
	{
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			Q[i][j] = Q_in.read();
		}
	}
	//rotation of rows: step1
	bool index = 0;
	for(uint_i j = ROWS - 1; j >= UNROLL_FACTOR; j--)
	{
//#pragma HLS unroll factor=2
#pragma HLS pipeline II=1
		uint_i k = j - UNROLL_FACTOR;
		C[index] = in_c.read();
		S[index] = in_s.read();
		qrf_mm(C[index], S[index], A[k], A[j]);
		out_c.write(C[index]);
		out_s.write(S[index]);
		index = ~index;
	}
	for(uint_i j = UNROLL_FACTOR - 1; j > 0; j--)
	{
		C[index] = in_c.read();
		S[index] = in_s.read();
		qrf_mm(C[index], S[index], A[j - 1], A[j]);
		out_c.write(C[index]);
		out_s.write(S[index]);
	}
	//rotation of rows: step2
	for(uint_i i = 1; i < id; i++)
	{
		for(uint_i j = ROWS - 1; j > i; j--)
		{
#pragma HLS pipeline II=1
			C[index] = in_c.read();
			S[index] = in_s.read();
			qrf_mm(C[index], S[index], A[j - UNROLL_FACTOR], A[j]);
			out_c.write(C[index]);
			out_s.write(S[index]);
			index = ~index;
		}
	}

	//angle generation
	for(uint_i i = ROWS - 1; i > id; i--)
	{
#pragma HLS pipeline II=1
		if(hls::abs(A[i]) < 1e-6)
		{
			C[index] = 1;
			S[index] = 0;
			for(uint_i k = 0; k < ROWS; k++)
			{
				qrf_mm(C[index], S[index], Q[k][i - UNROLL_FACTOR], Q[k][i]);
			}
		}
		else
		{
			MATRIX_T mag = qrf_mag(A[i], A[i - UNROLL_FACTOR]);
			C[index] = A[i - UNROLL_FACTOR] / mag;
			S[index] = A[i] / mag;
			A[i - UNROLL_FACTOR] = mag;
			A[i] = 0;
			for(uint_i k = 0; k < ROWS; k++)
			{
				qrf_mm(C[index], S[index], Q[k][i - UNROLL_FACTOR], Q[k][i]);
			}
		}
		out_c.write(C[index]);
		out_s.write(S[index]);
		index = ~index;
	}
	for(uint_i i = 0; i < ROWS; i++)
	{
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			Q_out.write(Q[i][j]);
		}
	}
	for(uint_i i = 0; i < id; i++)
	{
		for(uint_i i = 0; i < ROWS; i++)
		{
#pragma HLS pipeline II=1
			out_R.write(in_R.read());
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
		stream<MATRIX_T> &in_R,
		stream<MATRIX_T> &out_R,
		stream<MATRIX_T> &Q_in,
		stream<MATRIX_T> &Q_out)
{
	MATRIX_T A[ROWS];
	MATRIX_T C[2], S[2];
	MATRIX_T Q[ROWS][ROWS];

	//for operand A
	for(uint_i j = 0 ; j < ROWS; j++)
	{
#pragma HLS pipeline II=1
		A[j] = in_A.read();
	}
	for(uint_i i = 0; i < ROWS; i++)
	{
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			Q[i][j] = Q_in.read();
		}
	}
	//rotation of rows: step1
	bool index = 0;
	for(uint_i j = ROWS - 1; j >= UNROLL_FACTOR; j--)
	{
#pragma HLS pipeline II=1
		uint_i k = j - UNROLL_FACTOR;
		C[index] = in_c.read();
		S[index] = in_s.read();
		qrf_mm(C[index], S[index], A[k], A[j]);
		index = ~index;
	}
	for(uint_i j = UNROLL_FACTOR - 1; j > 0; j--)
	{
		C[index] = in_c.read();
		S[index] = in_s.read();
		qrf_mm(C[index], S[index], A[j - 1], A[j]);
	}
	//rotation of rows: step2
	for(uint_i i = 1; i < COLS - 1; i++)
	{
		for(uint_i j = ROWS - 1; j > i; j--)
		{
#pragma HLS pipeline II=1
			C[index] = in_c.read();
			S[index] = in_s.read();
			qrf_mm(C[index], S[index], A[j - UNROLL_FACTOR], A[j]);
			index = ~index;
		}
	}

	//angle generation
	for(uint_i i = ROWS - 1; i >= COLS; i--)
	{
#pragma HLS pipeline II=1
		if(hls::abs(A[i]) < 1e-6)
		{
			C[index] = 1;
			S[index] = 0;
			for(uint_i k = 0; k < ROWS; k++)
			{
				qrf_mm(C[index], S[index], Q[k][i - UNROLL_FACTOR], Q[k][i]);
			}
		}
		else
		{
			MATRIX_T mag = qrf_mag(A[i], A[i - UNROLL_FACTOR]);
			C[index] = A[i - UNROLL_FACTOR] / mag;
			S[index] = A[i] / mag;
			A[i - UNROLL_FACTOR] = mag;
			A[i] = 0;
			for(uint_i k = 0; k < ROWS; k++)
			{
				qrf_mm(C[index], S[index], Q[k][i - UNROLL_FACTOR], Q[k][i]);
			}
		}
		index = ~index;
	}
	for(uint_i i = 0; i < ROWS; i++)
	{
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			Q_out.write(Q[i][j]);
		}
	}
	for(uint_i i = 0; i < COLS - 1; i++)
	{
		for(uint_i i = 0; i < ROWS; i++)
		{
#pragma HLS pipeline II=1
			out_R.write(in_R.read());
		}
	}
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		out_R.write(A[i]);
	}
}

void collector(stream<MATRIX_T> &Q_in,
		stream<MATRIX_T> &in_R,
		MATRIX_T Q[ROWS][ROWS],
		MATRIX_T R[ROWS][COLS])
{
	//initialize for Q
	for(uint_i i = 0; i < ROWS; i++)
	{
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			Q[i][j] = Q_in.read();
		}
	}

	for(uint_i i = 0; i < COLS; i++)
	{
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			R[j][i] = in_R.read();
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
//; for($i = 0; $i < $COL - 1; $i++)
//; {
	stream<MATRIX_T> out_A`$i`;
#pragma HLS stream variable=out_A`$i` depth=10
//; }
//; for($i = 0; $i < $COL - 1; $i++)
//; {
	stream<MATRIX_T> pass_c`$i`;
#pragma HLS stream variable=pass_c`$i` depth=10
//; }
//; for($i = 0; $i < $COL - 1; $i++)
//; {
	stream<MATRIX_T> pass_s`$i`;
#pragma HLS stream variable=pass_s`$i` depth=10
//; }
//; for($i = 0; $i < $COL; $i++)
//; {
	stream<MATRIX_T> pass_Q`$i`;
#pragma HLS stream variable=pass_Q`$i` depth=10
//; }
//; for($i = 0; $i < $COL; $i++)
//; {
	stream<MATRIX_T> out_R`$i`;
#pragma HLS stream variable=out_R`$i` depth=10
//; }

	feeder(A, feedin);
	PE_head(feedin, out_A0, pass_c0, pass_s0, out_R0, pass_Q0);
//; $index = 1;
//; for($index = 1; $index < $COL - 1; $index++)
//; {
//; $index_1 = $index - 1;
	PE(`$index`, out_A`$index_1`, out_A`$index`, pass_c`$index_1`, pass_s`$index_1`, pass_c`$index`, pass_s`$index`, out_R`$index_1`, out_R`$index`, pass_Q`$index_1`, pass_Q`$index`);
//; }
//; $index = $index - 1;
//; $index_p1 = $index + 1;
	PE_tail(out_A`$index`, pass_c`$index`, pass_s`$index`, out_R`$index`, out_R`$index_p1`, pass_Q`$index`, pass_Q`$index_p1`);
	collector(pass_Q`$index_p1`, out_R`$index_p1`, Q, R);

	return 0;
}
