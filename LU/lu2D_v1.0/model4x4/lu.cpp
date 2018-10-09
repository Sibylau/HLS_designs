//This is a 2D systolic array implementation computing LU decomposition of 4x4 matrix A
//using Doolittle method with partial pivoting along projection vector (0,1,0).

#include "lu.h"
#include <iostream>

using namespace std;

//Feeder: feed in operands of each column in matrix A respectively from the bottom boundary of systolic array
void feeder(matrix_t A[matrix_size][matrix_size],
		stream<matrix_t> &feedin0,
		stream<matrix_t> &feedin1,
		stream<matrix_t> &feedin2,
		stream<matrix_t> &feedin3)
{
#pragma HLS array_partition variable=A complete dim=2
	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		feedin0.write(A[i][0]);
	}

	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		feedin1.write(A[i][1]);
	}

	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		feedin2.write(A[i][2]);
	}

	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		feedin3.write(A[i][3]);
	}

}

//id starts from 0
//The 1st type of PE (diagonal PE):
void PE1(uint_i id,
	stream<matrix_t> &feedin,//input: operands in matrix A
	stream<matrix_t> &L,//output final results in matrix L to the collector
	stream<matrix_t> &L_pass,//output final results in matrix U to the next PE
	stream<matrix_t> &U,//output final results in matrix U to the collector
	stream<uint_i> &P,//output pivoting information to the collector
	stream<uint_i> &P_pass)//output pivoting information to the next PE
{
	matrix_t maxpwr;
	uint_i pivot = id;
	matrix_t col[matrix_size];

//read in operands
	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		col[i] = feedin.read();
	}

//find the pivot element in one column
	maxpwr = hls::abs(col[id]);
	for(uint_i i = id + 1; i < matrix_size; i++)
	{
		if(maxpwr < hls::abs(col[i]))
		{
			maxpwr = hls::abs(col[i]);
			pivot = i;
		}
	}

	P.write(pivot);
	P_pass.write(pivot);
	if(pivot != id)//row exchange
	{
		maxpwr = col[pivot];
		col[pivot] = col[id];
		col[id] = maxpwr;
	}

//calculate the final results of matrix L and U
	matrix_t diag = col[id];
	for(uint_i i = id + 1; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		col[i] = col[i] / diag;
	}

	for(uint_i i = 0; i < id + 1; i++)
	{
#pragma HLS pipeline II=1
		U.write(col[i]);
	}

	for(uint_i i = id + 1; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		L.write(col[i]);
		L_pass.write(col[i]);
	}
}

//The 2nd type pf PE (non-diagonal PE):
void PE2(uint_i idx,//x coordinate of PE2
	uint_i idy,//y coordinate of PE2
	stream<matrix_t> &in,//input from the downward PE: middle results
	stream<matrix_t> &in_L,//input from the left PE: L(i,j)
	stream<matrix_t> &out_L,//output to the right PE: L(i,j)
	stream<uint_i> &in_P,//input from the left PE: P(i)
	stream<uint_i> &out_P,//output to the right PE: P(i)
	stream<matrix_t> &out)//output to the upward PE: middle results
{
	matrix_t col[matrix_size];
	matrix_t L_temp;
	uint_i pivot = in_P.read();

//read in operands:
	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		col[i] = in.read();
	}

//row exchange:
	L_temp = col[pivot];
	col[pivot] = col[idy];
	col[idy] = L_temp;
	out_P.write(pivot);

//calculate middle results:
	matrix_t col_temp = col[idy];
	for(uint_i i = idy + 1; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		L_temp = in_L.read();
		col[i] = col[i] - col_temp * L_temp;
		out_L.write(L_temp);
	}

//write out middle results:
	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		out.write(col[i]);
	}
}

//PE2_tail differs from PE2 in that it has no 'out_L' and 'out_P'.
void PE2_tail(uint_i idx,
		uint_i idy,
		stream<matrix_t> &in,
		stream<matrix_t> &in_L,
		stream<uint_i> &in_P,
		stream<matrix_t> &out)
{
	matrix_t col[matrix_size];
	matrix_t L_temp;
	uint_i pivot = in_P.read();

	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		col[i] = in.read();
	}

	//row exchange
	L_temp = col[pivot];
	col[pivot] = col[idy];
	col[idy] = L_temp;

	matrix_t col_temp = col[idy];
	for(uint_i i = idy + 1; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		L_temp = in_L.read();
		col[i] = col[i] - col_temp * L_temp;
	}

	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		out.write(col[i]);
	}
}

//collects all final results in matrix L and U, and pivoting information in P
void collector(
		stream<matrix_t> &out_L0,
		stream<matrix_t> &out_L1,
		stream<matrix_t> &out_L2,
		stream<matrix_t> &out_U0,
		stream<matrix_t> &out_U1,
		stream<matrix_t> &out_U2,
		stream<matrix_t> &out_U3,
		stream<uint_i> &P0,
		stream<uint_i> &P1,
		stream<uint_i> &P2,
		matrix_t L[matrix_size][matrix_size],
		matrix_t U[matrix_size][matrix_size],
		uint_i P[matrix_size])
{
	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		if(i == 0)
		{
			L[i][0] = 1;
			U[i][0] = out_U0.read();
		}
		else
		{
			L[i][0] = out_L0.read();
			U[i][0] = 0;
		}
	}

	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		if(i > 1)
		{
			L[i][1] = out_L1.read();
			U[i][1] = 0;
		}
		else if(i == 1)
		{
			L[i][1] = 1;
			U[i][1] = out_U1.read();
		}
		else
		{
			L[i][1] = 0;
			U[i][1] = out_U1.read();
		}
	}

	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		if(i > 2)
		{
			L[i][2] = out_L2.read();
			U[i][2] = 0;
		}
		else if(i == 2)
		{
			L[i][2] = 1;
			U[i][2] = out_U2.read();
		}
		else
		{
			L[i][2] = 0;
			U[i][2] = out_U2.read();
		}
	}

	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		if(i == 3)
		{
			L[i][3] = 1;
			U[i][3] = out_U3.read();
		}
		else
		{
			L[i][3] = 0;
			U[i][3] = out_U3.read();
		}
	}

	P[0] = P0.read();
	P[1] = P1.read();
	P[2] = P2.read();
	P[3] = 3;
}

int top(matrix_t A[matrix_size][matrix_size],
	    matrix_t L[matrix_size][matrix_size],
	    matrix_t U[matrix_size][matrix_size],
	    uint_i P[matrix_size])
{
#pragma HLS dataflow
	stream<matrix_t> out_L0;
#pragma HLS stream variable=out_L0 depth=10
	stream<matrix_t> out_L1;
#pragma HLS stream variable=out_L1 depth=10
	stream<matrix_t> out_L2;
#pragma HLS stream variable=out_L2 depth=10

	stream<matrix_t> out_U0;
#pragma HLS stream variable=out_U0 depth=10
	stream<matrix_t> out_U1;
#pragma HLS stream variable=out_U1 depth=10
	stream<matrix_t> out_U2;
#pragma HLS stream variable=out_U2 depth=10
	stream<matrix_t> out_U3;
#pragma HLS stream variable=out_U3 depth=10

	stream<matrix_t> mid_out0_1;
#pragma HLS stream variable=mid_out0_1 depth=10
	stream<matrix_t> mid_out0_2;
#pragma HLS stream variable=mid_out0_2 depth=10
	stream<matrix_t> mid_out0_3;
#pragma HLS stream variable=mid_out0_3 depth=10
	stream<matrix_t> mid_out1_1;
#pragma HLS stream variable=mid_out1_1 depth=10
	stream<matrix_t> mid_out1_2;
#pragma HLS stream variable=mid_out1_2 depth=10

	stream<uint_i> P0;
#pragma HLS stream variable=P0 depth=10
	stream<uint_i> P1;
#pragma HLS stream variable=P1 depth=10
	stream<uint_i> P2;
#pragma HLS stream variable=P2 depth=10

	stream<matrix_t> L0_1;
#pragma HLS stream variable=L0_1 depth=10
	stream<matrix_t> L0_2;
#pragma HLS stream variable=L0_2 depth=10
	stream<matrix_t> L0_3;
#pragma HLS stream variable=L0_3 depth=10
	stream<matrix_t> L1_1;
#pragma HLS stream variable=L1_1 depth=10
	stream<matrix_t> L1_2;
#pragma HLS stream variable=L1_2 depth=10
	stream<matrix_t> L2_1;
#pragma HLS stream variable=L2_1 depth=10

	stream<uint_i> P0_1;
#pragma HLS stream variable=P0_1 depth=10
	stream<uint_i> P0_2;
#pragma HLS stream variable=P0_2 depth=10
	stream<uint_i> P0_3;
#pragma HLS stream variable=P0_3 depth=10
	stream<uint_i> P1_1;
#pragma HLS stream variable=P1_1 depth=10
	stream<uint_i> P1_2;
#pragma HLS stream variable=P1_2 depth=10
	stream<uint_i> P2_1;
#pragma HLS stream variable=P2_1 depth=10

	stream<matrix_t> feedin0;
#pragma HLS stream variable=feedin0 depth=20
	stream<matrix_t> feedin1;
#pragma HLS stream variable=feedin1 depth=20
	stream<matrix_t> feedin2;
#pragma HLS stream variable=feedin2 depth=20
	stream<matrix_t> feedin3;
#pragma HLS stream variable=feedin3 depth=20
	
	feeder(A,
		feedin0,
		feedin1,
		feedin2,
	       	feedin3);

	PE1(0, feedin0, out_L0, L0_1, out_U0, P0, P0_1);
	PE2(1, 0, feedin1, L0_1, L0_2, P0_1, P0_2, mid_out0_1);
	PE2(2, 0, feedin2, L0_2, L0_3, P0_2, P0_3, mid_out0_2);
	PE2_tail(3, 0, feedin3, L0_3, P0_3, mid_out0_3);

	PE1(1, mid_out0_1, out_L1, L1_1, out_U1, P1, P1_1);
	PE2(2, 1, mid_out0_2, L1_1, L1_2, P1_1, P1_2, mid_out1_1);
	PE2_tail(3, 1, mid_out0_3, L1_2, P1_2, mid_out1_2);

	PE1(2, mid_out1_1, out_L2, L2_1, out_U2, P2, P2_1);
	PE2_tail(3, 2, mid_out1_2, L2_1, P2_1, out_U3);

	collector(
		out_L0,
		out_L1,
		out_L2,
	       	out_U0,
	       	out_U1,
	       	out_U2,
	       	out_U3,
	       	P0,
	       	P1,
	       	P2,
	       	L, U, P);
	return 0;
}
