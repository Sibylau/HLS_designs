//; $SIZE = param_define("SIZE", 4);
//; $SIZE_1 = $SIZE - 1;
//; $SUM = ($SIZE + 1) * $SIZE / 2;
//; $SUM_1 = $SIZE * ($SIZE - 1) / 2;
//; $SUM_Lv = ($SIZE - 1) * ($SIZE - 2) / 2;
#include "chol.h"
#include <iostream>

using namespace std;
void feeder(matrix_t A[matrix_size][matrix_size],
//; for($i = 0; $i < $SIZE_1; $i++)
//; {
	stream<matrix_t> &feedin`$i`,
//; }
	stream<matrix_t> &feedin`$SIZE_1`)
{
#pragma HLS array_partition variable=A complete dim=1
	int j;
	feedin0.write(A[0][0]);
//; for($i = 1; $i < $SIZE; $i++)
//; {
	for(j = `$i`; j >= 0; j--)
	{
#pragma HLS pipeline II=1
		feedin`$i`.write(A[`$i`][j]);
	}
//; }
}

//id start from 0
void collector(
//; for($i = 0; $i < $SIZE; $i++)
//; {
		 stream<matrix_t> &feedout`$i`,
//; }
		 matrix_t L[matrix_size][matrix_size])
{
	ap_uint<iterator_bit> j;
	
#pragma HLS array_partition variable=L complete dim=1
//; for($i = 0; $i < $SIZE_1; $i++)
//; {
	for(j = 0; j < matrix_size; j++)
	{
#pragma HLS pipeline II=1
		if(j > `$i`)
			L[`$i`][j] = 0;
		else
			L[`$i`][j] = feedout`$i`.read();
	}
//; }
	for(j = 0; j < matrix_size; j++)
	{
#pragma HLS pipeline II=1
		L[`$SIZE_1`][j] = feedout`$SIZE_1`.read();
	}
}

void PE1(ap_uint<iterator_bit> idx,
 	 ap_uint<iterator_bit> idy,
	 stream<matrix_t> &in_A,
	 stream<matrix_t> &in_Lh,
	 stream<matrix_t> &out_Lv,
	 stream<matrix_t> &output)
{
	ap_uint<iterator_bit> i;
	matrix_t A, L, Lh;

	//init
	A = in_A.read();

	//square, substract
	for(i = 0; i < idy; i++)
	{
		Lh = in_Lh.read();
		A = A - Lh * Lh;
		out_Lv.write(Lh);
		output.write(Lh);
	}

	//square root
	A = x_sqrt(A);
	out_Lv.write(A);
	output.write(A);
}

void PE1_tail(ap_uint<iterator_bit> idx,
		ap_uint<iterator_bit> idy,
		 stream<matrix_t> &in_A,
		 stream<matrix_t> &in_Lh,
		 stream<matrix_t> &output)
{
	ap_uint<iterator_bit> i;
	matrix_t A, L, Lh;

	//init
	A = in_A.read();

	//square, substract
	for(i = 0; i < idy; i++)
	{
		Lh = in_Lh.read();
		A = A - Lh * Lh;
		output.write(Lh);
	}

	//square root
	A = x_sqrt(A);
	output.write(A);
}

void PE2(ap_uint<iterator_bit> idx,
	ap_uint<iterator_bit> idy,
	stream<matrix_t> &in_A,
	stream<matrix_t> &out_A,
	stream<matrix_t> &in_Lh,
	stream<matrix_t> &out_Lh,//add an element in Lh
	stream<matrix_t> &in_Lv,
	stream<matrix_t> &out_Lv)
{
	ap_uint<iterator_bit> i;
	matrix_t A, Lh, Lv;

	//init
	for(i = 0; i < (idx - idy); i++)
	{
		out_A.write(in_A.read());
	}
	A = in_A.read();

	//multiply, subtract
	for(i = 0; i < idy; i++)
	{
		Lh = in_Lh.read();
		Lv = in_Lv.read();
		A = A - Lh * Lv;
		out_Lh.write(Lh);
		out_Lv.write(Lv);
	}

	//div
	Lv = in_Lv.read();
	out_Lv.write(Lv);

	A = A / Lv;
	out_Lh.write(A);
}

void PE2_tail(ap_uint<iterator_bit> idx,
		ap_uint<iterator_bit> idy,
		stream<matrix_t> &in_A,
		stream<matrix_t> &out_A,
		stream<matrix_t> &in_Lh,
		stream<matrix_t> &out_Lh,//add an element in Lh
		stream<matrix_t> &in_Lv)
{
	ap_uint<iterator_bit> i;
	matrix_t A, Lh, Lv;

	//init
	for(i = 0; i < (idx - idy); i++)
	{
		out_A.write(in_A.read());
	}
	A = in_A.read();

	//multiply, subtract
	for(i = 0; i < idy; i++)
	{
		Lh = in_Lh.read();
		Lv = in_Lv.read();
		A = A - Lh * Lv;
		out_Lh.write(Lh);
	}

	//div
	Lv = in_Lv.read();

	A = A / Lv;
	out_Lh.write(A);
}

int top(matrix_t A[matrix_size][matrix_size],
	matrix_t L[matrix_size][matrix_size])
{
	#pragma HLS DATAFLOW
//; for($i = 0; $i < $SUM_Lv; $i++)
//; {
	stream<matrix_t> Lv`$i`;
#pragma HLS STREAM variable=Lv`$i` depth=10
//; }

//; for($i = 0; $i < $SUM_1; $i++)
//; {
	stream<matrix_t> in_A`$i`;
#pragma HLS STREAM variable=in_A`$i` depth=10
//; }

//; for($i = 0; $i < $SUM; $i++)
//; {
	stream<matrix_t> Lh`$i`;
#pragma HLS STREAM variable=Lh`$i` depth=10
//; }

//; for($i = 0; $i < $SIZE; $i++)
//; {
	stream<matrix_t> Lii`$i`;
#pragma HLS STREAM variable=Lii`$i` depth=10
//; }

//; for($i = 0; $i < $SIZE; $i++)
//; {
	stream<matrix_t> feedin`$i`;
#pragma HLS STREAM variable=feedin`$i` depth=10
//; }

//; for($i = 0; $i < $SIZE; $i++)
//; {
	stream<matrix_t> feedout`$i`;
#pragma HLS STREAM variable=feedout`$i` depth=10
//; }

	feeder(A, 
//;for($i = 0; $i < $SIZE_1; $i++)
//; {
		feedin`$i`,
//; }
		feedin`$SIZE_1`);

//; $lv_count = 0;
//; $lv_count_p1 = $lv_count + 1;
//; $in_A_count = 1;
//; $lh_count = 2;
//; $increment = $SIZE_1;
	PE1(0, 0, feedin0, Lh0, Lii0, feedout0);
	PE2(1, 0, feedin1, in_A0, Lh1, Lh`$SIZE`, Lii0, Lv0);
//;for($i = 2; $i < $SIZE_1; $i++)
//;{
//; $lh_out = $lh_count + $increment;
	PE2(`$i`, 0, feedin`$i`, in_A`$in_A_count`, Lh`$lh_count`, Lh`$lh_out`, Lv`$lv_count`, Lv`$lv_count_p1`);
//; $lv_count = $lv_count + 1;
//; $lv_count_p1 = $lv_count + 1;
//; $in_A_count = $in_A_count + 1;
//; $lh_count = $lh_count + 1;
//;}
//; $lh_out = $lh_count + $increment;
	PE2_tail(`$SIZE_1`, 0, feedin`$SIZE_1`, in_A`$in_A_count`, Lh`$lh_count`, Lh`$lh_out`, Lv`$lv_count`);

//;for($i = 1; $i < $SIZE_1; $i++)
//;{
//; $increment = $increment - 1;
//; $in_A_behind = $in_A_count - $increment;
//; $lh_count = $lh_count + 1;
//; $i_p1 = $i + 1;
	PE1(`$i`, `$i`, in_A`$in_A_behind`, Lh`$lh_count`, Lii`$i`, feedout`$i`);
//; $in_A_behind = $in_A_behind + 1;
//; $in_A_count = $in_A_count + 1;
//; $lh_count = $lh_count + 1;
//; $lh_out = $lh_count + $increment;
//; $lv_count = $lv_count + 1;
//; if($i == $SIZE_1 - 1)
//; {
	PE2_tail(`$i_p1`, `$i`, in_A`$in_A_behind`, in_A`$in_A_count`, Lh`$lh_count`, Lh`$lh_out`, Lii`$i`);

//; }
//; else
//; {
	PE2(`$i_p1`, `$i`, in_A`$in_A_behind`, in_A`$in_A_count`, Lh`$lh_count`, Lh`$lh_out`, Lii`$i`, Lv`$lv_count`);
//; for($j = $i + 2; $j < $SIZE_1; $j++)
//; {
//; $in_A_behind = $in_A_behind + 1;
//; $in_A_count = $in_A_count + 1;
//; $lh_count = $lh_count + 1;
//; $lh_out = $lh_count + $increment;
//; $lv_count_p1 = $lv_count + 1;
	PE2(`$j`, `$i`, in_A`$in_A_behind`, in_A`$in_A_count`, Lh`$lh_count`, Lh`$lh_out`, Lv`$lv_count`, Lv`$lv_count_p1`);
//; $lv_count = $lv_count + 1;
//; }
//; $in_A_behind = $in_A_behind + 1;
//; $in_A_count = $in_A_count + 1;
//; $lh_count = $lh_count + 1;
//; $lh_out = $lh_count + $increment;
	PE2_tail(`$SIZE_1`, `$i`, in_A`$in_A_behind`, in_A`$in_A_count`, Lh`$lh_count`, Lh`$lh_out`, Lv`$lv_count`);

//; }
//;}
	PE1_tail(`$SIZE_1`, `$SIZE_1`, in_A`$in_A_count`, Lh`$lh_out`, feedout`$SIZE_1`);

	collector(
//; for($i = 0; $i < $SIZE; $i++)
//; {
		feedout`$i`,
//; }
	       	L);
	return 0;
}

