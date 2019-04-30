#include "chol.h"
#include <iostream>

using namespace std;
void feeder(matrix_t A[matrix_size][matrix_size],
	stream<matrix_t> &feedin0,
	stream<matrix_t> &feedin1)
{
#pragma HLS array_partition variable=A complete dim=1
	int j;
	feedin0.write(A[0][0]);
	for(j = 1; j >= 0; j--)
	{
#pragma HLS pipeline II=1
		feedin1.write(A[1][j]);
	}
}

//id start from 0
void collector(
		 stream<matrix_t> &feedout0,
		 stream<matrix_t> &feedout1,
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
		L[1][j] = feedout1.read();
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

	stream<matrix_t> in_A0;
#pragma HLS STREAM variable=in_A0 depth=10

	stream<matrix_t> Lh0;
#pragma HLS STREAM variable=Lh0 depth=10
	stream<matrix_t> Lh1;
#pragma HLS STREAM variable=Lh1 depth=10
	stream<matrix_t> Lh2;
#pragma HLS STREAM variable=Lh2 depth=10

	stream<matrix_t> Lii0;
#pragma HLS STREAM variable=Lii0 depth=10
	stream<matrix_t> Lii1;
#pragma HLS STREAM variable=Lii1 depth=10

	stream<matrix_t> feedin0;
#pragma HLS STREAM variable=feedin0 depth=10
	stream<matrix_t> feedin1;
#pragma HLS STREAM variable=feedin1 depth=10

	stream<matrix_t> feedout0;
#pragma HLS STREAM variable=feedout0 depth=10
	stream<matrix_t> feedout1;
#pragma HLS STREAM variable=feedout1 depth=10

	feeder(A, 
		feedin0,
		feedin1);

	PE1(0, 0, feedin0, Lh0, Lii0, feedout0);
	PE2(1, 0, feedin1, in_A0, Lh1, Lh2, Lii0, Lv0);
	PE2_tail(1, 0, feedin1, in_A1, Lh2, Lh3, Lv0);

	PE1_tail(1, 1, in_A1, Lh3, feedout1);

	collector(
		feedout0,
		feedout1,
	       	L);
	return 0;
}

