//This is a 2D systolic array implementation of QR decomposition
//using Givens Rotations along projection vector (1,0,0).

#include "qr.h"
#include <iostream>

using namespace std;

//Feeder: feed in operands of each column in matrix A respectively from the bottom boundary of systolic array
void feeder(MATRIX_T A[ROWS][COLS],
	stream<MATRIX_T> &feedin0,
	stream<MATRIX_T> &feedin1,
	stream<MATRIX_T> &feedin2,
	stream<MATRIX_T> &feedin3)
{
#pragma HLS array_partition variable=A complete dim=2
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		feedin0.write(A[i][0]);
	}
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		feedin1.write(A[i][1]);
	}
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		feedin2.write(A[i][2]);
	}
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		feedin3.write(A[i][3]);
	}
}

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

//The 1st type of PE (diagonal PE):
void PE1(uint_i id,
	stream<MATRIX_T> &in,
	stream<MATRIX_T> &pass_c,//output cosine values to PEs on the right
	stream<MATRIX_T> &pass_s,//output sine values to PEs on the right
	stream<MATRIX_T> &out_c,//output cosine values to the collector
	stream<MATRIX_T> &out_s,//output sine values to the collector
	stream<MATRIX_T> &out_R)//output final results in R to the collector
{
	MATRIX_T A[ROWS];

//read in operands from the first column of A:
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		A[i] = in.read();
	}

//Angle Generation:
	for(uint_i i = ROWS - 1; i > id; i--)
	{
#pragma HLS loop_flatten off
		if(A[i] < 1e-6)
		{
			MATRIX_T c = 1;
			MATRIX_T s = 0;
			pass_c.write(c);
			out_c.write(c);
			pass_s.write(s);
			out_s.write(s);
		}
		else
		{
			MATRIX_T mag = qrf_mag(A[i], A[i - 1]);
			MATRIX_T c = A[i - 1] / mag;
			MATRIX_T s = A[i] / mag;
			A[i - 1] = mag;
			A[i] = 0;
			pass_c.write(c);
			out_c.write(c);
			pass_s.write(s);
			out_s.write(s);
		}
	}

//output final results in R:
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		out_R.write(A[i]);
	}
}

//PE1_tail is located at the right boundary of systolic array, 
//and does not need to pass cosine and sine values to PEs on the right.
void PE1_tail(
		stream<MATRIX_T> &in,
		stream<MATRIX_T> &out_c,
		stream<MATRIX_T> &out_s,
		stream<MATRIX_T> &out_R)
{
	MATRIX_T A[ROWS];

//read in operands from the first column of A:
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		A[i] = in.read();
	}

//Angle Generation:
	for(uint_i i = ROWS - 1; i >= COLS; i--)
	{
#pragma HLS loop_flatten off
		if(A[i] == 0)
		{
			MATRIX_T c = 1;
			MATRIX_T s = 0;
			out_c.write(c);
			out_s.write(s);
		}
		else
		{
			MATRIX_T mag = qrf_mag(A[i], A[i - 1]);
			MATRIX_T c = A[i - 1] / mag;
			MATRIX_T s = A[i] / mag;
			A[i - 1] = mag;
			A[i] = 0;
			out_c.write(c);
			out_s.write(s);
		}
	}

//output final results in R:
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		out_R.write(A[i]);
	}
}

//The 2nd type pf PE (non-diagonal PE):
void PE2(uint_i idx,//x coordinate of PE2
	uint_i idy,//y coordinate of PE2
	stream<MATRIX_T> &in,//input from the PE below, or the feeder
	stream<MATRIX_T> &in_c,//input cosine values from PEs on the left
	stream<MATRIX_T> &in_s,//input sine values from PEs on the left
	stream<MATRIX_T> &out_c,//output cosine values to PEs on the right
	stream<MATRIX_T> &out_s,//output sine values to PEs on the right
	stream<MATRIX_T> &out_mid)//output middle results to the upward PE
{
	MATRIX_T A[ROWS];

//read in the operands of matrix A:
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		A[i] = in.read();
	}

//row exchange:
	for(uint_i i = ROWS - 1; i > idx; i--)
	{
		MATRIX_T c = in_c.read();
		MATRIX_T s = in_s.read();
		qrf_mm(c, s, A[i - 1], A[i]);
		out_c.write(c);
		out_s.write(s);
	}

//output middle results:
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		out_mid.write(A[i]);
	}
}

//PE2_tail is located at the right boundary of systolic array, 
//and does not need to pass cosine and sine values to PEs on the right.
void PE2_tail(
		uint_i idx,
		stream<MATRIX_T> &in,
		stream<MATRIX_T> &in_c,
		stream<MATRIX_T> &in_s,
		stream<MATRIX_T> &out_mid)
{
	MATRIX_T A[ROWS];

//read in the operands of matrix A:
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		A[i] = in.read();
	}

//row exchange:
	for(uint_i i = ROWS - 1; i > idx; i--)
	{
		MATRIX_T c = in_c.read();
		MATRIX_T s = in_s.read();
		qrf_mm(c, s, A[i - 1], A[i]);
	}

//output middle results:
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		out_mid.write(A[i]);
	}
}

void collector(
		stream<MATRIX_T> &in_c0,
		stream<MATRIX_T> &in_c1,
		stream<MATRIX_T> &in_c2,
		stream<MATRIX_T> &in_c3,
		stream<MATRIX_T> &in_s0,
		stream<MATRIX_T> &in_s1,
		stream<MATRIX_T> &in_s2,
		stream<MATRIX_T> &in_s3,
		stream<MATRIX_T> &in_R0,
		stream<MATRIX_T> &in_R1,
		stream<MATRIX_T> &in_R2,
		stream<MATRIX_T> &in_R3,
		MATRIX_T Q[ROWS][ROWS],
		MATRIX_T R[ROWS][COLS])
{
	MATRIX_T Q_i[ROWS][ROWS];

//initialize for Q and R:
#pragma HLS array_partition variable=Q_i cyclic factor=2 dim=2
	for(uint_i i = 0; i < ROWS; i++)
	{
		#pragma HLS loop_merge force
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
		for(uint_i j = 0; j < COLS; j++)
		{
			#pragma HLS pipeline II=1
			R[i][j] = 0;
		}
	}

//Rotations on matrix Q:
	for(uint_i i = ROWS - 1; i > 0; i--)
	{
		MATRIX_T c = in_c0.read();
		MATRIX_T s = in_s0.read();
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			qrf_mm(c, s, Q_i[j][i - 1], Q_i[j][i]);
		}
	}
//read in R:
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		R[i][0] = in_R0.read();
	}

//Rotations on matrix Q:
	for(uint_i i = ROWS - 1; i > 1; i--)
	{
		MATRIX_T c = in_c1.read();
		MATRIX_T s = in_s1.read();
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			qrf_mm(c, s, Q_i[j][i - 1], Q_i[j][i]);
		}
	}
//read in R:
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		R[i][1] = in_R1.read();
	}

//Rotations on matrix Q:
	for(uint_i i = ROWS - 1; i > 2; i--)
	{
		MATRIX_T c = in_c2.read();
		MATRIX_T s = in_s2.read();
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			qrf_mm(c, s, Q_i[j][i - 1], Q_i[j][i]);
		}
	}
//read in R:
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		R[i][2] = in_R2.read();
	}

//Rotations on matrix Q:
	for(uint_i i = ROWS - 1; i > 3; i--)
	{
		MATRIX_T c = in_c3.read();
		MATRIX_T s = in_s3.read();
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			qrf_mm(c, s, Q_i[j][i - 1], Q_i[j][i]);
		}
	}
//read in R:
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		R[i][3] = in_R3.read();
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
	//check QR operant
	if(ROWS < COLS)
	{
	#ifndef __SYNTHESIS__
        printf("ERROR: Parameter error - RowsA must be greater than ColsA; currently RowsA = %d ColsA = %d\n",ROWS,COLS);
	#endif
        exit(1);	
	}

	#pragma HLS dataflow

	stream<MATRIX_T> feedin0;
#pragma HLS stream variable=feedin0 depth=10
	stream<MATRIX_T> feedin1;
#pragma HLS stream variable=feedin1 depth=10
	stream<MATRIX_T> feedin2;
#pragma HLS stream variable=feedin2 depth=10
	stream<MATRIX_T> feedin3;
#pragma HLS stream variable=feedin3 depth=10

	stream<MATRIX_T> pass_c0_0;
#pragma HLS stream variable=pass_c0_0 depth=10
	stream<MATRIX_T> pass_c0_1;
#pragma HLS stream variable=pass_c0_1 depth=10
	stream<MATRIX_T> pass_c0_2;
#pragma HLS stream variable=pass_c0_2 depth=10
	stream<MATRIX_T> pass_c1_0;
#pragma HLS stream variable=pass_c1_0 depth=10
	stream<MATRIX_T> pass_c1_1;
#pragma HLS stream variable=pass_c1_1 depth=10
	stream<MATRIX_T> pass_c2_0;
#pragma HLS stream variable=pass_c2_0 depth=10

	stream<MATRIX_T> pass_s0_0;
#pragma HLS stream variable=pass_s0_0 depth=10
	stream<MATRIX_T> pass_s0_1;
#pragma HLS stream variable=pass_s0_1 depth=10
	stream<MATRIX_T> pass_s0_2;
#pragma HLS stream variable=pass_s0_2 depth=10
	stream<MATRIX_T> pass_s1_0;
#pragma HLS stream variable=pass_s1_0 depth=10
	stream<MATRIX_T> pass_s1_1;
#pragma HLS stream variable=pass_s1_1 depth=10
	stream<MATRIX_T> pass_s2_0;
#pragma HLS stream variable=pass_s2_0 depth=10

	stream<MATRIX_T> out_c0;
#pragma HLS stream variable=out_c0 depth=10
	stream<MATRIX_T> out_c1;
#pragma HLS stream variable=out_c1 depth=10
	stream<MATRIX_T> out_c2;
#pragma HLS stream variable=out_c2 depth=10
	stream<MATRIX_T> out_c3;
#pragma HLS stream variable=out_c3 depth=10

	stream<MATRIX_T> out_s0;
#pragma HLS stream variable=out_s0 depth=10
	stream<MATRIX_T> out_s1;
#pragma HLS stream variable=out_s1 depth=10
	stream<MATRIX_T> out_s2;
#pragma HLS stream variable=out_s2 depth=10
	stream<MATRIX_T> out_s3;
#pragma HLS stream variable=out_s3 depth=10

	stream<MATRIX_T> out_R0;
#pragma HLS stream variable=out_R0 depth=10
	stream<MATRIX_T> out_R1;
#pragma HLS stream variable=out_R1 depth=10
	stream<MATRIX_T> out_R2;
#pragma HLS stream variable=out_R2 depth=10
	stream<MATRIX_T> out_R3;
#pragma HLS stream variable=out_R3 depth=10

	stream<MATRIX_T> mid0_1;
#pragma HLS stream variable=mid0_1 depth=10
	stream<MATRIX_T> mid0_2;
#pragma HLS stream variable=mid0_2 depth=10
	stream<MATRIX_T> mid0_3;
#pragma HLS stream variable=mid0_3 depth=10
	stream<MATRIX_T> mid1_2;
#pragma HLS stream variable=mid1_2 depth=10
	stream<MATRIX_T> mid1_3;
#pragma HLS stream variable=mid1_3 depth=10
	stream<MATRIX_T> mid2_3;
#pragma HLS stream variable=mid2_3 depth=10

	feeder(A, 
		feedin0,
		feedin1,
		feedin2,
		feedin3);

	PE1(0, feedin0, pass_c0_0, pass_s0_0, out_c0, out_s0, out_R0);
	PE2(0, 1, feedin1, pass_c0_0, pass_s0_0, pass_c0_1, pass_s0_1, mid0_1);
	PE2(0, 2, feedin2, pass_c0_1, pass_s0_1, pass_c0_2, pass_s0_2, mid0_2);
	PE2_tail(0, feedin3, pass_c0_2, pass_s0_2, mid0_3);

	PE1(1, mid0_1, pass_c1_0, pass_s1_0, out_c1, out_s1, out_R1);
	PE2(1, 1, mid0_2, pass_c1_0, pass_s1_0, pass_c1_1, pass_s1_1, mid1_2);
	PE2_tail(1, mid0_3, pass_c1_1, pass_s1_1, mid1_3);
	PE1(2, mid1_2, pass_c2_0, pass_s2_0, out_c2, out_s2, out_R2);
	PE2_tail(2, mid1_3, pass_c2_0, pass_s2_0, out_R3);
//if (ROWS > COLS) 
//  the last PE1 is needed;
//if (ROWS == COLS)
//  the last PE1 will be omitted;


	collector(
		out_c0,
		out_c1,
		out_c2,
		out_c3,
	       	out_s0,
	       	out_s1,
	       	out_s2,
	       	out_s3,
	       	out_R0,
	       	out_R1,
	       	out_R2,
	       	out_R3,
		Q, R);
	return 0;
}
