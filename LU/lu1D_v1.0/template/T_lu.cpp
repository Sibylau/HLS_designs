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

void PE(uint_i id,
		stream<matrix_t> &in,
		stream<matrix_t> &out_L,
		stream<matrix_t> &out_U,
		stream<matrix_t> &mid,
		stream<uint_i> &P)
{
	matrix_t A[matrix_size][matrix_size];
	matrix_t M[matrix_size][matrix_size];
	matrix_t temp[matrix_size];
	uint_i pivot = id;
	matrix_t maxpwr;

	for(uint_i i = id; i < matrix_size; i++)
	{
		for(uint_i j = id; j < matrix_size; j++)
		{
#pragma HLS pipeline II=1
			A[j][i] = in.read();
		}
	}

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
		for(uint_i i = id; i < matrix_size; i++)
		{
#pragma HLS pipeline II=1
			temp[i] = A[pivot][i];
			A[pivot][i] = A[id][i];
			A[id][i] = temp[i];
		}
	}

	matrix_t diag = A[id][id];
	for(uint_i i = id + 1; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		A[i][id] = A[i][id] / diag;
		out_L.write(A[i][id]);
	}

	for(uint_i i = id; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		out_U.write(A[id][i]);
	}

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


void collector(
//; for($i = 0 ; $i < $SIZE_1; $i++)
//; {
		stream<matrix_t> &out_L`$i`,
//; }
//; for($i = 0; $i < $SIZE; $i++)
//; {
		stream<matrix_t> &out_U`$i`,
//; }
//; for($i = 0; $i < $SIZE_1; $i++)
//; {
		stream<uint_i> &P`$i`,
//; }
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

//; for($i = 1; $i < $SIZE_1; $i++)
//; {
	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		if(i > `$i`)
		{
			L[i][`$i`] = out_L`$i`.read();
		}
		else if(i == `$i`)
		{
			L[i][`$i`] = 1;
		}
		else
		{
			L[i][`$i`] = 0;
		}
	}
	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		if(i >= `$i`)
		{
			U[`$i`][i] = out_U`$i`.read();
		}
		else
		{
			U[`$i`][i] = 0;
		}
	}

//; }


	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		if(i == `$SIZE_1`)
		{
			L[i][`$SIZE_1`] = 1;
		}
		else
		{
			L[i][`$SIZE_1`] = 0;
			U[`$SIZE_1`][i] = 0;
		}
	}
	U[`$SIZE_1`][`$SIZE_1`] = out_U`$SIZE_1`.read();

//; for($i = 0; $i < $SIZE_1; $i++)
//; {
	P[`$i`] = P`$i`.read();
//; }
	P[`$SIZE_1`] = `$SIZE_1`;
}

int top(matrix_t A[matrix_size][matrix_size],
    matrix_t L[matrix_size][matrix_size],
    matrix_t U[matrix_size][matrix_size],
    uint_i P[matrix_size])
{
#pragma HLS dataflow

	stream<matrix_t> feedin;
#pragma HLS stream variable=feedin depth=10

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

//; for($i = 0; $i < $SIZE_1 - 1; $i++)
//; {
	stream<matrix_t> mid`$i`;
#pragma HLS stream variable=mid`$i` depth=10
//; }

//; for($i = 0; $i < $SIZE_1; $i++)
//; {
	stream<uint_i> P`$i`;
#pragma HLS stream variable=P`$i` depth=10
//; }
	
	feeder(A, feedin);
	PE(0, feedin, out_L0, out_U0, mid0, P0);
//; for($i = 1; $i < $SIZE_1 - 1; $i++)
//; {
//; $i_1 = $i - 1;
	PE(`$i`, mid`$i_1`, out_L`$i`, out_U`$i`, mid`$i`, P`$i`);
//; }
//; $last = $SIZE - 2;
//; $last_p1 = $last + 1;
//; $last_1 = $last - 1;
	PE(`$last`, mid`$last_1`, out_L`$last`, out_U`$last`, out_U`$last_p1`, P`$last`);

	collector(
//; for($i = 0; $i < $SIZE_1; $i++)
//; {
		out_L`$i`,
//; }
//; for($i = 0; $i < $SIZE; $i++)
//; {	
		out_U`$i`,
//; }
//; for($i = 0; $i < $SIZE_1; $i++)
//; {
	       	P`$i`,
//; }
	       	L, U, P);
	return 0;
}
