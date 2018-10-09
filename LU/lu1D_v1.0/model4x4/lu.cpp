//This is a 1D systolic array implementation performing LU decomposition of 4x4 matrix A
//using Doolittle method with partial pivoting along (0,1,0) and (0,1).

#include "lu.h"
#include <iostream>

using namespace std;

//Feeder: feed in the elements of matrix A in column major order.
void feeder(matrix_t A[matrix_size][matrix_size],
			stream<matrix_t> &feedin)
{
	for(uint_i i = 0; i < matrix_size; i++)
	{
		for(uint_i j = 0; j < matrix_size; j++)
		{
#pragma HLS pipeline II=1
			feedin.write(A[j][i]);
		}
	}
}

void PE(uint_i id,
		stream<matrix_t> &in,//input: operands
		stream<matrix_t> &out_L,//output: final results of matrix L
		stream<matrix_t> &out_U,//output: final results of matrix U
		stream<matrix_t> &mid,//output: middle results to the next PE
		stream<uint_i> &P)//output: the row number of the pivot element
{
	matrix_t A[matrix_size][matrix_size];
	matrix_t M[matrix_size][matrix_size];
	matrix_t temp[matrix_size];
	uint_i pivot = id;
	matrix_t maxpwr;

//read in operands of matrix A
	for(uint_i i = id; i < matrix_size; i++)
	{
		for(uint_i j = id; j < matrix_size; j++)
		{
#pragma HLS pipeline II=1
			A[j][i] = in.read();
		}
	}

//find the pivot element within one column
	maxpwr = hls::abs(A[id][id]);
	for(uint_i i = id + 1; i < matrix_size; i++)
	{
		if(maxpwr < hls::abs(A[i][id]))
		{
			maxpwr = hls::abs(A[i][id]);
			pivot = i;
		}
	}

	P.write(pivot);
	if(pivot != id)
	{
		for(uint_i i = id; i < matrix_size; i++)//exchange between the pivot row and the current row
		{
#pragma HLS pipeline II=1
			temp[i] = A[pivot][i];
			A[pivot][i] = A[id][i];
			A[id][i] = temp[i];
		}
	}

//calculate final results in matrix L
	matrix_t diag = A[id][id];
	for(uint_i i = id + 1; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		A[i][id] = A[i][id] / diag;
		out_L.write(A[i][id]);
	}

//calculate final results in matrix U
	for(uint_i i = id; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		out_U.write(A[id][i]);
	}

//update the values of middle results
	for(uint_i i = id + 1; i < matrix_size; i++)
	{
		for(uint_i j = id + 1; j < matrix_size; j++)
		{
#pragma HLS pipeline II=1
			M[j][i] = A[j][i];
			M[j][i] = M[j][i] - A[j][id] * A[id][i];
			mid.write(M[j][i]);
		}
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
		if(i > 0)
			L[i][0] = out_L0.read();
		else
			L[i][0] = 1;
	}
	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1	
		U[0][i] = out_U0.read();
	}

	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		if(i > 1)
		{
			L[i][1] = out_L1.read();
		}
		else if(i == 1)
		{
			L[i][1] = 1;
		}
		else
		{
			L[i][1] = 0;
		}
	}
	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		if(i >= 1)
		{
			U[1][i] = out_U1.read();
		}
		else
		{
			U[1][i] = 0;
		}
	}

	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		if(i > 2)
		{
			L[i][2] = out_L2.read();
		}
		else if(i == 2)
		{
			L[i][2] = 1;
		}
		else
		{
			L[i][2] = 0;
		}
	}
	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		if(i >= 2)
		{
			U[2][i] = out_U2.read();
		}
		else
		{
			U[2][i] = 0;
		}
	}



	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		if(i == 3)
		{
			L[i][3] = 1;
		}
		else
		{
			L[i][3] = 0;
			U[3][i] = 0;
		}
	}
	U[3][3] = out_U3.read();

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

	stream<matrix_t> feedin;
#pragma HLS stream variable=feedin depth=10

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

	stream<matrix_t> mid0;
#pragma HLS stream variable=mid0 depth=10
	stream<matrix_t> mid1;
#pragma HLS stream variable=mid1 depth=10

	stream<uint_i> P0;
#pragma HLS stream variable=P0 depth=10
	stream<uint_i> P1;
#pragma HLS stream variable=P1 depth=10
	stream<uint_i> P2;
#pragma HLS stream variable=P2 depth=10
	
	feeder(A, feedin);
	PE(0, feedin, out_L0, out_U0, mid0, P0);
	PE(1, mid0, out_L1, out_U1, mid1, P1);
	PE(2, mid1, out_L2, out_U2, out_U3, P2);

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
