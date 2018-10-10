//; $SIZE = param_define("SIZE", 6);
//; $SIZE_1 = $SIZE - 1;
//; $SIZE_2 = $SIZE - 2;
//This program computes LU decomposition of A
//using Doolittle method with partial pivoting.

#include "lu.h"
#include <iostream>

using namespace std;

void feeder(matrix_t A[matrix_size][matrix_size],
//; for($i = 0; $i < $SIZE_1; $i++)
//; {
		stream<matrix_t> &feedin`$i`,
//; }
		stream<matrix_t> &feedin`$SIZE_1`)
{
#pragma HLS array_partition variable=A complete dim=2
//; for($i = 0; $i < $SIZE; $i++)
//; {
	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		feedin`$i`.write(A[i][`$i`]);
	}

//; }
}

//id starts from 0
void PE1(uint_i id,
	stream<matrix_t> &feedin,
	stream<matrix_t> &L,
	stream<matrix_t> &L_pass,
	stream<matrix_t> &U,
	stream<uint_i> &P,
	stream<uint_i> &P_pass)
{
	matrix_t maxpwr;
	uint_i pivot = id;
	matrix_t col[matrix_size];

	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		col[i] = feedin.read();
	}
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
	if(pivot != id)
	{
		maxpwr = col[pivot];
		col[pivot] = col[id];
		col[id] = maxpwr;
	}

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

void PE2(uint_i idx,
	uint_i idy,
	stream<matrix_t> &in,
	stream<matrix_t> &in_L,
	stream<matrix_t> &out_L,
	stream<uint_i> &in_P,
	stream<uint_i> &out_P,
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

	//pivoting
	L_temp = col[pivot];
	col[pivot] = col[idy];
	col[idy] = L_temp;
	out_P.write(pivot);

	matrix_t col_temp = col[idy];
	for(uint_i i = idy + 1; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		L_temp = in_L.read();
		col[i] = col[i] - col_temp * L_temp;
		out_L.write(L_temp);
	}

	for(uint_i i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		out.write(col[i]);
	}
}

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

	//pivoting
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

void collector(
//; for($i = 0; $i < $SIZE_1; $i++)
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
		if(i == `$SIZE_1`)
		{
			L[i][`$SIZE_1`] = 1;
			U[i][`$SIZE_1`] = out_U`$SIZE_1`.read();
		}
		else
		{
			L[i][`$SIZE_1`] = 0;
			U[i][`$SIZE_1`] = out_U`$SIZE_1`.read();
		}
	}

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

//; for($i = 0; $i < $SIZE_2; $i++)
//; {
//;  for($j = 1; $j < $SIZE - $i; $j++)
//;  {
	stream<matrix_t> mid_out`$i`_`$j`;
#pragma HLS stream variable=mid_out`$i`_`$j` depth=10
//;  }
//; }

//; for($i = 0; $i < $SIZE_1; $i++)
//; {
	stream<uint_i> P`$i`;
#pragma HLS stream variable=P`$i` depth=10
//; }

//; for($i = 0; $i < $SIZE_1; $i++)
//; {
//;  for($j = 1; $j < $SIZE - $i; $j++)
//;  {
	stream<matrix_t> L`$i`_`$j`;
#pragma HLS stream variable=L`$i`_`$j` depth=10
//;  }
//; }

//; for($i = 0; $i < $SIZE_1; $i++)
//; {
//;  for($j = 1; $j < $SIZE - $i; $j++)
//;  {
	stream<uint_i> P`$i`_`$j`;
#pragma HLS stream variable=P`$i`_`$j` depth=10
//;  }
//; }

//; for($i = 0; $i < $SIZE; $i++)
//; {
	stream<matrix_t> feedin`$i`;
#pragma HLS stream variable=feedin`$i` depth=20
//; }
	
	feeder(A,
//; for($i = 0; $i < $SIZE_1; $i++)
//; {
		feedin`$i`,
//; }
	       	feedin`$SIZE_1`);

	PE1(0, feedin0, out_L0, L0_1, out_U0, P0, P0_1);
//; for($i = 1; $i < $SIZE_1; $i++)
//; {
//; $i_p1 = $i + 1;
	PE2(`$i`, 0, feedin`$i`, L0_`$i`, L0_`$i_p1`, P0_`$i`, P0_`$i_p1`, mid_out0_`$i`);
//; }
	PE2_tail(`$SIZE_1`, 0, feedin`$SIZE_1`, L0_`$SIZE_1`, P0_`$SIZE_1`, mid_out0_`$SIZE_1`);

//; for($i = 1; $i < $SIZE_1; $i++)
//; {
//; $i_1 = $i - 1;
//; $cnt = 2;
//; $cnt_1 = 1;
	PE1(`$i`, mid_out`$i_1`_1, out_L`$i`, L`$i`_1, out_U`$i`, P`$i`, P`$i`_1);
//;  for($j = $i + 1; $j < $SIZE_1; $j++)
//;  {
	PE2(`$j`, `$i`, mid_out`$i_1`_`$cnt`, L`$i`_`$cnt_1`, L`$i`_`$cnt`, P`$i`_`$cnt_1`, P`$i`_`$cnt`, mid_out`$i`_`$cnt_1`);
//;   $cnt = $cnt + 1;
//;   $cnt_1 = $cnt - 1;
//;  }
//; $mid_out_2 = $SIZE - $i;
//; if($i != $SIZE_2)
//; {
	PE2_tail(`$SIZE_1`, `$i`, mid_out`$i_1`_`$mid_out_2`, L`$i`_`$cnt_1`, P`$i`_`$cnt_1`, mid_out`$i`_`$cnt_1`);

//; }
//; else
//; {
	PE2_tail(`$SIZE_1`, `$i`, mid_out`$i_1`_`$mid_out_2`, L`$i`_1, P`$i`_1, out_U`$SIZE_1`);

//; }
//; }
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
