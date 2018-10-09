//This is a 1D systolic array implementation of QR decomposition
//using Givens Rotations along projection vector (1,0,0) and (0,1).

#include "qr.h"
#include <iostream>

using namespace std;

MATRIX_T qrf_mag(MATRIX_T a, MATRIX_T b)//computes the magnitude of a and b
{
#pragma HLS inline
	MATRIX_T aa = a * a;
	MATRIX_T bb = b * b;
	MATRIX_T mag = x_sqrt(aa + bb);
	return mag;
}

//qrf_mm: can be used for both left mm and right mm
//for left mm:
// [a b]|c -s|
//	|s  c|
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

//feeder: feed data in column major order
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
		stream<MATRIX_T> &out_A,//pass operands of A to the next PE
		stream<MATRIX_T> &pass_c,//pass cosine values to the next PE
		stream<MATRIX_T> &pass_s,//pass sine values to the next PE
		stream<MATRIX_T> &out_R)//output final results in R
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

//Angle Generation:
	//rotations between row (i-UNROLL_FACTOR) and (i)
	//Every two rotations are independent.
	//UNROLL_FACTOR is 2 by default.
	for(uint_i i = ROWS - 1; i >= UNROLL_FACTOR; i--)
	{
#pragma HLS unroll factor=2
//cannot be executed concurrently due to writing the same output FIFO.
		if(hls::abs(A[i]) < 1e-6) //rounding error
		{
			MATRIX_T c = 1;
			MATRIX_T s = 0;
			pass_c.write(c);
			pass_s.write(s);
		}
		else
		{
//to improve parallelism
//two neighboring rotations are independent when UNROLL_FACTOR=2
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
//rotations between column (i-1) and (i)
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

//output final results of the first column in A
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

//Rotation of rows:
//step1: rotations according to PE_head
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
//step2: rotations according to the successive PEs
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

//Angle Generation:
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

//output final results in R
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		out_R.write(A[i]);
	}
}
//PE_tail differs from PE in that it does not have FIFOs 'out_A'.
void PE_tail(
		stream<MATRIX_T> &in_A,
		stream<MATRIX_T> &in_c,
		stream<MATRIX_T> &in_s,
		stream<MATRIX_T> &out_c,
		stream<MATRIX_T> &out_s,
		stream<MATRIX_T> &out_R)
{
	MATRIX_T A[ROWS];

//read operands in A
	for(uint_i j = 0 ; j < ROWS; j++)
	{
#pragma HLS pipeline II=1
		A[j] = in_A.read();
	}

//Rotation of rows: 
//step1: rotations according to PE_head
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
//step2: rotations according to successive PEs.
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

//Angle Generation:
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

//output final results of matrix R
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		out_R.write(A[i]);
	}
}

void collector(stream<MATRIX_T> &in_c,
		stream<MATRIX_T> &in_s,
		stream<MATRIX_T> &in_R0,
		stream<MATRIX_T> &in_R1,
		stream<MATRIX_T> &in_R2,
		stream<MATRIX_T> &in_R3,
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
//read in R
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		R[i][0] = in_R0.read();
	}
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		R[i][1] = in_R1.read();
	}
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		R[i][2] = in_R2.read();
	}
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		R[i][3] = in_R3.read();
	}

//Rotations on matrix Q:
//step1: rotations according to PE_head
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
//step2: rotations according to successive PEs
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
//copy Q_i to Q, to reduce the latency of the collector in DATAFLOW
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

	stream<MATRIX_T> out_A0;
#pragma HLs stream variable=out_A0 depth=10
	stream<MATRIX_T> out_A1;
#pragma HLs stream variable=out_A1 depth=10
	stream<MATRIX_T> out_A2;
#pragma HLs stream variable=out_A2 depth=10

	stream<MATRIX_T> pass_c0;
#pragma HLS stream variable=pass_c0 depth=10
	stream<MATRIX_T> pass_c1;
#pragma HLS stream variable=pass_c1 depth=10
	stream<MATRIX_T> pass_c2;
#pragma HLS stream variable=pass_c2 depth=10
	stream<MATRIX_T> pass_c3;
#pragma HLS stream variable=pass_c3 depth=10

	stream<MATRIX_T> pass_s0;
#pragma HLS stream variable=pass_s0 depth=10
	stream<MATRIX_T> pass_s1;
#pragma HLS stream variable=pass_s1 depth=10
	stream<MATRIX_T> pass_s2;
#pragma HLS stream variable=pass_s2 depth=10
	stream<MATRIX_T> pass_s3;
#pragma HLS stream variable=pass_s3 depth=10

	stream<MATRIX_T> out_R0;
#pragma HLS stream variable=out_R0 depth=10
	stream<MATRIX_T> out_R1;
#pragma HLS stream variable=out_R1 depth=10
	stream<MATRIX_T> out_R2;
#pragma HLS stream variable=out_R2 depth=10
	stream<MATRIX_T> out_R3;
#pragma HLS stream variable=out_R3 depth=10

	feeder(A, feedin);
	PE_head(feedin, out_A0, pass_c0, pass_s0, out_R0);
	PE(1, out_A0, out_A1, pass_c0, pass_s0, pass_c1, pass_s1, out_R1);
	PE(2, out_A1, out_A2, pass_c1, pass_s1, pass_c2, pass_s2, out_R2);
	PE_tail(out_A2, pass_c2, pass_s2, pass_c3, pass_s3, out_R3);
	collector(pass_c3, pass_s3,
		 out_R0,
		 out_R1,
		 out_R2,
		 out_R3,
		 Q, R);
	return 0;
}
