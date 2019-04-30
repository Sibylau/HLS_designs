#include "chol.h"
#include <iostream>

using namespace std;
void feeder(matrix_t A[matrix_size][matrix_size],
	stream<matrix_t> &feedin0,
	stream<matrix_t> &feedin1,
	stream<matrix_t> &feedin2)
{
#pragma HLS array_partition variable=A complete dim=1
	int j;
	feedin0.write(A[0][0]);
	for(j = 1; j >= 0; j--)
	{
#pragma HLS pipeline II=1
		feedin1.write(A[1][j]);
	}
	for(j = 2; j >= 0; j--)
	{
#pragma HLS pipeline II=1
		feedin2.write(A[2][j]);
	}
}

//id start from 0
void collector(
		 stream<matrix_t> &feedout0,
		 stream<matrix_t> &feedout1,
		 stream<matrix_t> &feedout2,
		 matrix_t L[matrix_size][matrix_size])
{
	ap_uint<iterator_bit> j;
	
#pragma HLS array_partition variable=L complete dim=1
	for(j = 0; j < matrix_size; j++)
	{
#pragma HLS pipeline II=1
		if(j > 0)
			L[0][j] = 0;
		else
			L[0][j] = feedout0.read();
	}
	for(j = 0; j < matrix_size; j++)
	{
#pragma HLS pipeline II=1
		if(j > 1)
			L[1][j] = 0;
		else
			L[1][j] = feedout1.read();
	}
	for(j = 0; j < matrix_size; j++)
	{
#pragma HLS pipeline II=1
		L[2][j] = feedout2.read();
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
	stream<matrix_t> Lv0;
#pragma HLS STREAM variable=Lv0 depth=10

	stream<matrix_t> in_A0;
#pragma HLS STREAM variable=in_A0 depth=10
	stream<matrix_t> in_A1;
#pragma HLS STREAM variable=in_A1 depth=10
	stream<matrix_t> in_A2;
#pragma HLS STREAM variable=in_A2 depth=10

	stream<matrix_t> Lh0;
#pragma HLS STREAM variable=Lh0 depth=10
	stream<matrix_t> Lh1;
#pragma HLS STREAM variable=Lh1 depth=10
	stream<matrix_t> Lh2;
#pragma HLS STREAM variable=Lh2 depth=10
	stream<matrix_t> Lh3;
#pragma HLS STREAM variable=Lh3 depth=10
	stream<matrix_t> Lh4;
#pragma HLS STREAM variable=Lh4 depth=10
	stream<matrix_t> Lh5;
#pragma HLS STREAM variable=Lh5 depth=10

	stream<matrix_t> Lii0;
#pragma HLS STREAM variable=Lii0 depth=10
	stream<matrix_t> Lii1;
#pragma HLS STREAM variable=Lii1 depth=10
	stream<matrix_t> Lii2;
#pragma HLS STREAM variable=Lii2 depth=10

	stream<matrix_t> feedin0;
#pragma HLS STREAM variable=feedin0 depth=10
	stream<matrix_t> feedin1;
#pragma HLS STREAM variable=feedin1 depth=10
	stream<matrix_t> feedin2;
#pragma HLS STREAM variable=feedin2 depth=10

	stream<matrix_t> feedout0;
#pragma HLS STREAM variable=feedout0 depth=10
	stream<matrix_t> feedout1;
#pragma HLS STREAM variable=feedout1 depth=10
	stream<matrix_t> feedout2;
#pragma HLS STREAM variable=feedout2 depth=10

	feeder(A, 
		feedin0,
		feedin1,
		feedin2);

	PE1(0, 0, feedin0, Lh0, Lii0, feedout0);
	PE2(1, 0, feedin1, in_A0, Lh1, Lh3, Lii0, Lv0);
	PE2_tail(2, 0, feedin2, in_A1, Lh2, Lh4, Lv0);

	PE1(1, 1, in_A0, Lh3, Lii1, feedout1);
	PE2_tail(2, 1, in_A1, in_A2, Lh4, Lh5, Lii1);

	PE1_tail(2, 2, in_A2, Lh5, feedout2);

	collector(
		feedout0,
		feedout1,
		feedout2,
	       	L);
	return 0;
}

