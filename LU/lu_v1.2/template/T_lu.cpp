//; $SIZE = param_define("SIZE", 6);
//; $SIZE_1 = $SIZE - 1;
//This program computes LU decomposition of A
//using Doolittle method with partial pivoting.

#include "lu.h"
#include <iostream>

using namespace std;

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
		stream<matrix_t> &out_A,
		stream<matrix_t> &out_L,
		stream<matrix_t> &out_U,
		stream<matrix_t> &mid_L,
		stream<uint_i> &mid_P)
{
	matrix_t A[matrix_size];

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
	mid_P.write(pivot);

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

	for(uint_i i = id; i < matrix_size; i++)//id
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
 		if(pivot != i)
 		{
 			maxpwr = A[pivot];
 			A[pivot] = A[i];
 			A[i] = maxpwr;
 		}
 		out_mid_P.write(pivot);
 		temp_U = A[i];
 		out_U.write(A[i]);
 		for(uint_i j = 1; j < matrix_size; j++)
 		{
#pragma HLS loop_flatten off
#pragma HLS pipeline II=1
 			if(j > i)
 			{
 			L[j] = in_mid_L.read();
 			A[j] = A[j] - temp_U * L[j];
 			out_mid_L.write(L[j]);
 			}
 		}
 	}

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
 	out_mid_P.write(pivot);

	temp_U = A[id];
	out_U.write(temp_U);
	matrix_t temp_L;
	for(uint_i i = id+1; i < matrix_size; i++)//id+1
	{
#pragma HLS pipeline II=1
		temp_L = A[i] / temp_U;
		out_L.write(temp_L);
		out_mid_L.write(temp_L);
	}
}

void PE_tail(
		stream<matrix_t> &in_A,
		stream<matrix_t> &out_U,
		stream<matrix_t> &in_mid_L,
		stream<uint_i> &in_mid_P,
		stream<uint_i> &out_mid_P)
{
	matrix_t A[matrix_size];
	matrix_t L[matrix_size];
	matrix_t temp_U;

	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		A[i] = in_A.read();
	}

	uint_i pivot;
 	matrix_t maxpwr;

	for(uint_i i = 0; i < matrix_size - 1; i++)
	{
		pivot = in_mid_P.read();
		if(pivot != i)
		{
			maxpwr = A[pivot];
			A[pivot] = A[i];
			A[i] = maxpwr;
		}
		out_U.write(A[i]);
		out_mid_P.write(pivot);
		temp_U = A[i];
		for(uint_i j = 1; j < matrix_size; j++)//i+1
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
	out_U.write(temp_U);
}

void collector(
//; for($i = 0; $i < $SIZE_1; $i++)
//; {
		stream<matrix_t> &out_L`$i`,
//; }
//; for($i = 0; $i < $SIZE; $i++)
//; {
		stream<matrix_t> &out_U`$i`,
//; }
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

//; for($i = 1; $i < $SIZE_1; $i++)
//; {
	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		if(i > `$i`)
		{
			L[i][`$i`] = out_L`$i`.read();
			U[i][`$i`] = 0;
		}
		else if(i == `$i`)
		{
			L[i][`$i`] = 1;
			U[i][`$i`] = out_U`$i`.read();
		}
		else
		{
			L[i][`$i`] = 0;
			U[i][`$i`] = out_U`$i`.read();
		}
	}

//; }
	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		if(i < `$SIZE_1`)
		{
			L[i][`$SIZE_1`] = 0;
		}
		else
		{
			L[i][`$SIZE_1`] = 1;
		}
	}

	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		P[i] = out_P.read();
		U[i][`$SIZE_1`] = out_U`$SIZE_1`.read();
	}
}


int top(matrix_t A[matrix_size][matrix_size],
	    matrix_t L[matrix_size][matrix_size],
	    matrix_t U[matrix_size][matrix_size],
	    uint_i P[matrix_size])
{
#pragma HLS dataflow

//; for($i = 0; $i < $SIZE_1; $i++)
//; {
	stream<matrix_t> out_A`$i`;
#pragma HLS stream variable=out_A`$i` depth=10
//; }

//; for($i = 0; $i < $SIZE_1; $i++)
//; {
	stream<matrix_t> out_L`$i`;
#pragma HLS stream variable=out_L`$i` depth=10
//; }

//; for($i = 0; $i < $SIZE; $i++)
//; {
	stream<matrix_t> out_U`$i`;
#pragma HLS stream variable=out_U`$i` depth=10
//; }

//; for($i = 0; $i < $SIZE; $i++)
//; {
	stream<uint_i> mid_P`$i`;
#pragma HLS stream variable=mid_P`$i` depth=10
//; }

//; for($i = 0; $i < $SIZE_1; $i++)
//; {
	stream<matrix_t> mid_L`$i`;
#pragma HLS stream variable=mid_L`$i` depth=10
//; }
	stream<matrix_t> feedin;
#pragma HLS stream variable=feedin depth=10

	feeder(A, feedin);

	PE_head(feedin, out_A0, out_L0, out_U0, mid_L0, mid_P0);
//; for($i = 1; $i < $SIZE_1; $i++)
//; {
//; $i_1 = $i - 1;
	PE(`$i`, out_A`$i_1`, out_A`$i`, out_L`$i`, out_U`$i`, mid_L`$i_1`, mid_L`$i`, mid_P`$i_1`, mid_P`$i`);
//; }
//; $SIZE_2 = $SIZE - 2;
	PE_tail(out_A`$SIZE_2`, out_U`$SIZE_1`, mid_L`$SIZE_2`, mid_P`$SIZE_2`, mid_P`$SIZE_1`);
	collector(
//; for($i = 0; $i < $SIZE_1; $i++)
//; {
		out_L`$i`,
//; }
//; for($i = 0; $i < $SIZE; $i++)
//; {
	       	out_U`$i`,
//; }
	       	mid_P`$SIZE_1`, L, U, P);
	return 0;
}
