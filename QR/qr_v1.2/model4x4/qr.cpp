//This is a 1D systolic array implementation of QR decomposition
//using Givens Rotations with pipelined rotation generation 
//along projection vector (1,0,0) and (0,1).
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

//qrf_mm: can be used for both left and right matrix vector multiplication
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

//PE_head differs from PE in that it does not contain FIFOs 'in_c', 'in_s', 'in_R', and 'Q_in' 
void PE_head(stream<MATRIX_T> &in_A,
		stream<MATRIX_T> &out_A,//pass operands of A to the next PE
		stream<MATRIX_T> &pass_c,//pass cosine values to the next PE
		stream<MATRIX_T> &pass_s,//pass sine values to the next PE
		stream<MATRIX_T> &out_R,//output final results in R
		stream<MATRIX_T> &Q_out)//output middle results in Q
{
	MATRIX_T A[ROWS];
	MATRIX_T A_temp;
	MATRIX_T Q[ROWS][ROWS];
	MATRIX_T C[2];
	MATRIX_T S[2];
#pragma HLS array_partition variable=A complete
#pragma HLS array_partition variable=C complete
#pragma HLS array_partition variable=S complete
#pragma HLS array_partition variable=Q complete dim=2

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

//initialize for Q
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

//Angle Generation:
	bool index = 0;
	//rotations between row (i-UNROLL_FACTOR) and (i)
	//Every two rotations are independent.
	//UNROLL_FACTOR is 2 by default.
	for(uint_i i = ROWS - 1; i >= UNROLL_FACTOR; i--)
	{
#pragma HLS pipeline II=1
//Pipeline is used instead of UNROLL, as two independent rotations 
//cannot be executed concurrently due to writing the same output FIFO.
//II=1 cannot be achieved due to inevitable dependency of Q and A
		if(hls::abs(A[i]) < 1e-6) //rounding error
		{
			C[index] = 1;
			S[index] = 0;
			for(uint_i k = 0; k < ROWS; k++)
			{
				//multiply with a unit matrix when A[i] = 0
				qrf_mm(C[index], S[index], Q[k][i - UNROLL_FACTOR], Q[k][i]);
			}
		}
		else
		{
			//to improve parallelism
			//two neighboring rotations are independent when UNROLL_FACTOR=2
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

	//rotations between column (i-1) and (i), cannot be pipelined
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

//write out elements in Q
	for(uint_i i = 0; i < ROWS; i++)
	{
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			Q_out.write(Q[i][j]);
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
		stream<MATRIX_T> &in_R,
		stream<MATRIX_T> &out_R,
		stream<MATRIX_T> &Q_in,
		stream<MATRIX_T> &Q_out)
{
	MATRIX_T A[ROWS];
	MATRIX_T C[2], S[2];
	MATRIX_T Q[ROWS][ROWS];
#pragma HLS array_partition variable=A complete
#pragma HLS array_partition variable=C complete
#pragma HLS array_partition variable=S complete
#pragma HLS array_partition variable=Q complete dim=2

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

//read in the matrix Q
	for(uint_i i = 0; i < ROWS; i++)
	{
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			Q[i][j] = Q_in.read();
		}
	}

//Rotation of rows:
//step1: rotations according to PE_head
	bool index = 0;
	for(uint_i j = ROWS - 1; j >= UNROLL_FACTOR; j--)
	{
//#pragma HLS unroll factor=2
#pragma HLS pipeline II=1 
//cannot achieve due to dependent operations of A
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
//step2: rotations according to the successive PEs
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

//Angle Generation:
	for(uint_i i = ROWS - 1; i > id; i--)
	{
#pragma HLS pipeline II=1
//cannot achieve II=1 due to inevitable dependency of Q and A
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

//output middle results in Q
	for(uint_i i = 0; i < ROWS; i++)
	{
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			Q_out.write(Q[i][j]);
		}
	}

//output final results in R
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

//PE_tail differs from PE in that it does not have FIFOs 'out_A', 'out_c', 'out_s'.
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
#pragma HLS array_partition variable=A complete
#pragma HLS array_partition variable=C complete
#pragma HLS array_partition variable=S complete
#pragma HLS array_partition variable=Q complete dim=2

//read operands in A
	for(uint_i j = 0 ; j < ROWS; j++)
	{
#pragma HLS pipeline II=1
		A[j] = in_A.read();
	}

//read in the matrix Q
	for(uint_i i = 0; i < ROWS; i++)
	{
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			Q[i][j] = Q_in.read();
		}
	}

//Rotation of rows: 
//step1: rotations according to PE_head
	bool index = 0;
	for(uint_i j = ROWS - 1; j >= UNROLL_FACTOR; j--)
	{
#pragma HLS pipeline II=1 
//cannot achieve due to dependent operations of A
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

//step2: rotations according to successive PEs
	for(uint_i i = 1; i < COLS - 1; i++)
	{
		for(uint_i j = ROWS - 1; j > i; j--)
		{
#pragma HLS pipeline II=1 
//cannot achieve due to dependent operations of A
			C[index] = in_c.read();
			S[index] = in_s.read();
			qrf_mm(C[index], S[index], A[j - UNROLL_FACTOR], A[j]);
			index = ~index;
		}
	}

//Angle Generation:
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

//output final results of matrix Q
	for(uint_i i = 0; i < ROWS; i++)
	{
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			Q_out.write(Q[i][j]);
		}
	}

//output final results of matrix R
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
//read in Q
	for(uint_i i = 0; i < ROWS; i++)
	{
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			Q[i][j] = Q_in.read();
		}
	}
//read in R
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
	stream<MATRIX_T> out_A0;
#pragma HLS stream variable=out_A0 depth=10
	stream<MATRIX_T> out_A1;
#pragma HLS stream variable=out_A1 depth=10
	stream<MATRIX_T> out_A2;
#pragma HLS stream variable=out_A2 depth=10
	stream<MATRIX_T> pass_c0;
#pragma HLS stream variable=pass_c0 depth=10
	stream<MATRIX_T> pass_c1;
#pragma HLS stream variable=pass_c1 depth=10
	stream<MATRIX_T> pass_c2;
#pragma HLS stream variable=pass_c2 depth=10
	stream<MATRIX_T> pass_s0;
#pragma HLS stream variable=pass_s0 depth=10
	stream<MATRIX_T> pass_s1;
#pragma HLS stream variable=pass_s1 depth=10
	stream<MATRIX_T> pass_s2;
#pragma HLS stream variable=pass_s2 depth=10
	stream<MATRIX_T> pass_Q0;
#pragma HLS stream variable=pass_Q0 depth=10
	stream<MATRIX_T> pass_Q1;
#pragma HLS stream variable=pass_Q1 depth=10
	stream<MATRIX_T> pass_Q2;
#pragma HLS stream variable=pass_Q2 depth=10
	stream<MATRIX_T> pass_Q3;
#pragma HLS stream variable=pass_Q3 depth=10
	stream<MATRIX_T> out_R0;
#pragma HLS stream variable=out_R0 depth=10
	stream<MATRIX_T> out_R1;
#pragma HLS stream variable=out_R1 depth=10
	stream<MATRIX_T> out_R2;
#pragma HLS stream variable=out_R2 depth=10
	stream<MATRIX_T> out_R3;
#pragma HLS stream variable=out_R3 depth=10

	feeder(A, feedin);
	PE_head(feedin, out_A0, pass_c0, pass_s0, out_R0, pass_Q0);
	PE(1, out_A0, out_A1, pass_c0, pass_s0, pass_c1, pass_s1, out_R0, out_R1, pass_Q0, pass_Q1);
	PE(2, out_A1, out_A2, pass_c1, pass_s1, pass_c2, pass_s2, out_R1, out_R2, pass_Q1, pass_Q2);
	PE_tail(out_A2, pass_c2, pass_s2, out_R2, out_R3, pass_Q2, pass_Q3);
	collector(pass_Q3, out_R3, Q, R);

	return 0;
}
