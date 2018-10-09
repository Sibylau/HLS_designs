//This is a 2D systolic array implementation of Cholesky algorithm. 
#include "chol.h"
#include <iostream>

using namespace std;

//Feeder: feeds in data from the left boundary in row major order 
void feeder(matrix_t A[matrix_size][matrix_size],
	stream<matrix_t> &feedin0,
	stream<matrix_t> &feedin1,
	stream<matrix_t> &feedin2,
	stream<matrix_t> &feedin3)
{
#pragma HLS array_partition variable=A complete dim=1
	int j;
	feedin0.write(A[0][0]);
	for(j = 1; j >= 0; j--)//feed in the whole 2nd row
	{
#pragma HLS pipeline II=1
		feedin1.write(A[1][j]);
	}
	for(j = 2; j >= 0; j--)
	{
#pragma HLS pipeline II=1
		feedin2.write(A[2][j]);
	}
	for(j = 3; j >= 0; j--)
	{
#pragma HLS pipeline II=1
		feedin3.write(A[3][j]);
	}
}

//Collector: collect data from the right boundary in row major order
void collector(
		 stream<matrix_t> &feedout0,
		 stream<matrix_t> &feedout1,
		 stream<matrix_t> &feedout2,
		 stream<matrix_t> &feedout3,
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
		if(j > 2)
			L[2][j] = 0;
		else
			L[2][j] = feedout2.read();
	}
	for(j = 0; j < matrix_size; j++)
	{
#pragma HLS pipeline II=1
		L[3][j] = feedout3.read();
	}
}

//The 1st type of PE (diagonal PE): 
void PE1(ap_uint<iterator_bit> idx,//PE_id: x-axis
 	 ap_uint<iterator_bit> idy,//PE_id: y-axis
	 stream<matrix_t> &in_A,//input: operand A
	 stream<matrix_t> &in_Lh,//input: L(i,j), 0<j<i 
	 stream<matrix_t> &out_Lv,//output to off-diagonal PEs: L(i,i)
	 stream<matrix_t> &output)//output to the collector: L(i,i)
{
	ap_uint<iterator_bit> i;
	matrix_t A, L, Lh;

	//read in A
	A = in_A.read();

	//read in L(i,j), 0<j<i, substract squares of L(i,j) from A 
	for(i = 0; i < idy; i++)
	{
		Lh = in_Lh.read();
		A = A - Lh * Lh;
		out_Lv.write(Lh);//L(i,j) is also needed for PEs under itself
		output.write(Lh);//output final results L(i,j) to the collector
	}

	//square root
	A = x_sqrt(A);//get L(i,i)
	out_Lv.write(A);//output L(i,i) to off-diagonal PEs in the same column
	output.write(A);//output L(i,i) to the collector
}

//PE1_tails differs from PE1 only in that the interface FIFOs do not contain 'out_Lv'.
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

//The 2nd type of PE (off-diagonal PE):
void PE2(ap_uint<iterator_bit> idx,
	ap_uint<iterator_bit> idy,
	stream<matrix_t> &in_A,
	stream<matrix_t> &out_A,//pass the operands in A horizontally
	stream<matrix_t> &in_Lh,//get previous L(i,j) from PEs on the left
	stream<matrix_t> &out_Lh,//add a new element L(idx,idy) to Lh
	stream<matrix_t> &in_Lv,
	stream<matrix_t> &out_Lv)//pass L(i,i) downwards from diagonal PEs
{
	ap_uint<iterator_bit> i;
	matrix_t A, Lh, Lv;

	//pass the operands horizontally
	for(i = 0; i < (idx - idy); i++)
	{
		out_A.write(in_A.read());
	}
	A = in_A.read();//read in operand A

	//multiply L(i,j) and L(i,i), then subtract the product from A
	for(i = 0; i < idy; i++)
	{
		Lh = in_Lh.read();
		Lv = in_Lv.read();
		A = A - Lh * Lv;
		out_Lh.write(Lh);
		out_Lv.write(Lv);
	}

	//division
	Lv = in_Lv.read();
	out_Lv.write(Lv);

	A = A / Lv;//obtain L(i,j)
	out_Lh.write(A);//pass L(i,j) to PEs on the right
}

//PE2_tail differs from PE2 in that the interface FIFOs do not contain 'out_Lv'.
void PE2_tail(ap_uint<iterator_bit> idx,
		ap_uint<iterator_bit> idy,
		stream<matrix_t> &in_A,
		stream<matrix_t> &out_A,
		stream<matrix_t> &in_Lh,
		stream<matrix_t> &out_Lh,
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
#pragma HLS STREAM variable=Lv0 depth=10//to avoid potential FIFO stalls
	stream<matrix_t> Lv1;
#pragma HLS STREAM variable=Lv1 depth=10
	stream<matrix_t> Lv2;
#pragma HLS STREAM variable=Lv2 depth=10

	stream<matrix_t> in_A0;
#pragma HLS STREAM variable=in_A0 depth=10
	stream<matrix_t> in_A1;
#pragma HLS STREAM variable=in_A1 depth=10
	stream<matrix_t> in_A2;
#pragma HLS STREAM variable=in_A2 depth=10
	stream<matrix_t> in_A3;
#pragma HLS STREAM variable=in_A3 depth=10
	stream<matrix_t> in_A4;
#pragma HLS STREAM variable=in_A4 depth=10
	stream<matrix_t> in_A5;
#pragma HLS STREAM variable=in_A5 depth=10

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
	stream<matrix_t> Lh6;
#pragma HLS STREAM variable=Lh6 depth=10
	stream<matrix_t> Lh7;
#pragma HLS STREAM variable=Lh7 depth=10
	stream<matrix_t> Lh8;
#pragma HLS STREAM variable=Lh8 depth=10
	stream<matrix_t> Lh9;
#pragma HLS STREAM variable=Lh9 depth=10

	stream<matrix_t> Lii0;
#pragma HLS STREAM variable=Lii0 depth=10
	stream<matrix_t> Lii1;
#pragma HLS STREAM variable=Lii1 depth=10
	stream<matrix_t> Lii2;
#pragma HLS STREAM variable=Lii2 depth=10
	stream<matrix_t> Lii3;
#pragma HLS STREAM variable=Lii3 depth=10

	stream<matrix_t> feedin0;
#pragma HLS STREAM variable=feedin0 depth=10
	stream<matrix_t> feedin1;
#pragma HLS STREAM variable=feedin1 depth=10
	stream<matrix_t> feedin2;
#pragma HLS STREAM variable=feedin2 depth=10
	stream<matrix_t> feedin3;
#pragma HLS STREAM variable=feedin3 depth=10

	stream<matrix_t> feedout0;
#pragma HLS STREAM variable=feedout0 depth=10
	stream<matrix_t> feedout1;
#pragma HLS STREAM variable=feedout1 depth=10
	stream<matrix_t> feedout2;
#pragma HLS STREAM variable=feedout2 depth=10
	stream<matrix_t> feedout3;
#pragma HLS STREAM variable=feedout3 depth=10

	feeder(A, 
		feedin0,
		feedin1,
		feedin2,
		feedin3);
//first column
	PE1(0, 0, feedin0, Lh0, Lii0, feedout0);
	PE2(1, 0, feedin1, in_A0, Lh1, Lh4, Lii0, Lv0);
	PE2(2, 0, feedin2, in_A1, Lh2, Lh5, Lv0, Lv1);
	PE2_tail(3, 0, feedin3, in_A2, Lh3, Lh6, Lv1);
//second column
	PE1(1, 1, in_A0, Lh4, Lii1, feedout1);
	PE2(2, 1, in_A1, in_A3, Lh5, Lh7, Lii1, Lv2);
	PE2_tail(3, 1, in_A2, in_A4, Lh6, Lh8, Lv2);
//third column
	PE1(2, 2, in_A3, Lh7, Lii2, feedout2);
	PE2_tail(3, 2, in_A4, in_A5, Lh8, Lh9, Lii2);
//last column
	PE1_tail(3, 3, in_A5, Lh9, feedout3);

	collector(
		feedout0,
		feedout1,
		feedout2,
		feedout3,
	       	L);
	return 0;
}

