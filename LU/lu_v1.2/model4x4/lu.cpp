//This is a 1D systolic array implementation performing LU decomposition of A
//using Doolittle method with partial pivoting along (0,1,0) and (1,0).

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

void PE_head(stream<matrix_t> &in_A,
		stream<matrix_t> &out_A,//pass operands in matrix A to the next PE
		stream<matrix_t> &out_L,//output final results in matrix L to the collector
		stream<matrix_t> &out_U,//output final results in matrix U to the collector
		stream<matrix_t> &mid_L,//pass middle results to the next PE
		stream<uint_i> &mid_P)//pass pivoting information to the next PE 
{
	matrix_t A[matrix_size];

//loop: read in and pass operands of matrix A
	for(uint_i i = 0; i < matrix_size; i++) 
	{
		for(uint_i j = 0; j < matrix_size; j++)
		{
#pragma HLS pipeline II=1
			if(i == 0)
			{
				A[j] = in_A.read();
			}
			else
			{
				out_A.write(in_A.read());
			}
		}
	}

//find the pivot element in one column
	matrix_t maxpwr = hls::abs(A[0]);
	uint_i pivot = 0;

	for(uint_i i = 1; i < matrix_size; i++)
	{
		if(hls::abs(A[i]) > maxpwr)
		{
			maxpwr = hls::abs(A[i]);
			pivot = i;
		}
	}
	if(pivot != 0)
	{
		maxpwr = A[pivot];
		A[pivot] = A[0];
		A[0] = maxpwr;
	}
	mid_P.write(pivot);//pass the row number of pivot element to the next PE

//calculate final results in L and U, and pass middle results to the next PE
	matrix_t diag_A = A[0];
	out_U.write(A[0]);
	for(uint_i i = 1; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		A[i] = A[i] / diag_A;
		out_L.write(A[i]);
		mid_L.write(A[i]);
	}
}
 
//id starts from 1
void PE(uint_i id,
	stream<matrix_t> &in_A,
	stream<matrix_t> &out_A,
	stream<matrix_t> &out_L,
	stream<matrix_t> &out_U,
	stream<matrix_t> &in_mid_L,
	stream<matrix_t> &out_mid_L,
	stream<uint_i> &in_mid_P,
	stream<uint_i> &out_mid_P)
{
	matrix_t A[matrix_size];
	matrix_t A1[matrix_size];
	matrix_t L[matrix_size];
	matrix_t temp_U;

//loop: read in and pass operands of matrix A
	for(uint_i i = id; i < matrix_size; i++)
	{
		for(uint_i j = 0; j < matrix_size; j++)
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
 
 	uint_i pivot;
 	matrix_t maxpwr;

 	for(uint_i i = 0; i < id; i++)
 	{
 		pivot = in_mid_P.read();
 		if(pivot != i)//row exchange according to the pivot information
 		{
 			maxpwr = A[pivot];
 			A[pivot] = A[i];
 			A[i] = maxpwr;
 		}
 		out_mid_P.write(pivot);
 		temp_U = A[i];
 		out_U.write(A[i]);
 		for(uint_i j = 1; j < matrix_size; j++)//update according to previous results
 		{
#pragma HLS loop_flatten off
#pragma HLS pipeline II=1
 			if(j > i)
 			{
 			L[j] = in_mid_L.read();
 			A[j] = A[j] - temp_U * L[j];
 			out_mid_L.write(L[j]);//previous results of L are still used by successive PEs
 			}
 		}
 	}

//find the pivot element in one column
 	pivot = id;
 	maxpwr = hls::abs(A[id]);
 	for(uint_i i = id + 1; i < matrix_size; i++)//id+1
 	{
 		if(maxpwr < hls::abs(A[i]))
 		{
 			maxpwr = hls::abs(A[i]);
 			pivot = i;
 		}
 	}
 	if(pivot != id)
 	{
 		maxpwr = A[pivot];
 		A[pivot] = A[id];
 		A[id] = maxpwr;
 	}
 	out_mid_P.write(pivot);//add a new pivot to the queue

//calculate final results in L and U, and pass middle results to the next PE
	temp_U = A[id];
	out_U.write(temp_U);
	matrix_t temp_L;
	for(uint_i i = id+1; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		temp_L = A[i] / temp_U;
		out_L.write(temp_L);
		out_mid_L.write(temp_L);
	}
}

void PE_tail(
		stream<matrix_t> &in_A,
		stream<matrix_t> &out_U,//no 'out_L' in Doolittle method
		stream<matrix_t> &in_mid_L,
		stream<uint_i> &in_mid_P,
		stream<uint_i> &out_mid_P)
{
	matrix_t A[matrix_size];
	matrix_t L[matrix_size];
	matrix_t temp_U;

	for(uint_i i = 0; i < matrix_size; i++)//read in operands
	{
#pragma HLS pipeline II=1
		A[i] = in_A.read();
	}

	uint_i pivot;
 	matrix_t maxpwr;

	for(uint_i i = 0; i < matrix_size - 1; i++)
	{
		pivot = in_mid_P.read();
		if(pivot != i)//row exchange according to pivot information
		{
			maxpwr = A[pivot];
			A[pivot] = A[i];
			A[i] = maxpwr;
		}
		out_U.write(A[i]);
		out_mid_P.write(pivot);
		temp_U = A[i];
		for(uint_i j = 1; j < matrix_size; j++)//update according to previous results
		{
#pragma HLS loop_flatten off
#pragma HLS pipeline II=1
			if(j > i)
			{
			L[j] = in_mid_L.read();
			A[j] = A[j] - temp_U * L[j];
			}
		}
	}
	pivot = matrix_size - 1;
	out_mid_P.write(pivot);
	temp_U = A[matrix_size - 1];
	out_U.write(temp_U);//the last final result in matrix U
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
		stream<uint_i> &out_P,
		matrix_t L[matrix_size][matrix_size],
		matrix_t U[matrix_size][matrix_size],
		uint_i P[matrix_size])
{
	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		if(i > 0)
		{
			L[i][0] = out_L0.read();
			U[i][0] = 0;
		}
		else
		{
			L[i][0] = 1;
			U[i][0] = out_U0.read();
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
		if(i < 3)
		{
			L[i][3] = 0;
		}
		else
		{
			L[i][3] = 1;
		}
	}

	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		P[i] = out_P.read();
		U[i][3] = out_U3.read();
	}
}


int top(matrix_t A[matrix_size][matrix_size],
	    matrix_t L[matrix_size][matrix_size],
	    matrix_t U[matrix_size][matrix_size],
	    uint_i P[matrix_size])
{
#pragma HLS dataflow

	stream<matrix_t> out_A0;
#pragma HLS stream variable=out_A0 depth=10
	stream<matrix_t> out_A1;
#pragma HLS stream variable=out_A1 depth=10
	stream<matrix_t> out_A2;
#pragma HLS stream variable=out_A2 depth=10

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

	stream<uint_i> mid_P0;
#pragma HLS stream variable=mid_P0 depth=10
	stream<uint_i> mid_P1;
#pragma HLS stream variable=mid_P1 depth=10
	stream<uint_i> mid_P2;
#pragma HLS stream variable=mid_P2 depth=10
	stream<uint_i> mid_P3;
#pragma HLS stream variable=mid_P3 depth=10

	stream<matrix_t> mid_L0;
#pragma HLS stream variable=mid_L0 depth=10
	stream<matrix_t> mid_L1;
#pragma HLS stream variable=mid_L1 depth=10
	stream<matrix_t> mid_L2;
#pragma HLS stream variable=mid_L2 depth=10
	stream<matrix_t> feedin;
#pragma HLS stream variable=feedin depth=10

	feeder(A, feedin);

	PE_head(feedin, out_A0, out_L0, out_U0, mid_L0, mid_P0);
	PE(1, out_A0, out_A1, out_L1, out_U1, mid_L0, mid_L1, mid_P0, mid_P1);
	PE(2, out_A1, out_A2, out_L2, out_U2, mid_L1, mid_L2, mid_P1, mid_P2);
	PE_tail(out_A2, out_U3, mid_L2, mid_P2, mid_P3);
	collector(
		out_L0,
		out_L1,
		out_L2,
	       	out_U0,
	       	out_U1,
	       	out_U2,
	       	out_U3,
	       	mid_P3, L, U, P);
	return 0;
}
