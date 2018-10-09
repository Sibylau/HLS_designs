#include "chol.h"
//This is a 1D systolic array implementation of Cholesky algorithm
//along (0,1,0) and (1,0).
#include <iostream>

using namespace std;

//Feeder: feed in the elements in the lower triangular matrix of A
//   	  (Note that A is a symmetric matrix) in column major order.
void feeder(matrix_t A[matrix_size][matrix_size],
			stream<matrix_t> &feedin)
{
	for(ap_uint<iterator_bit> i = 0; i < matrix_size; i++)
	{
		for(ap_uint<iterator_bit> j = 0; j < matrix_size; j++)
		{
#pragma HLS pipeline II=1
			if(j >= i)
				feedin.write(A[j][i]);
		}
	}
}

//PE_head differs from PE in that it does not contain FIFOs 'in_A', 'in_col' and 'in_row'. 
void PE_head(stream<matrix_t> &in_A,
		stream<matrix_t> &out_A,//pass operand A to the next PE
		stream<matrix_t> &out_col,//multiplier 1: shared in the same column
		stream<matrix_t> &out_row,//multiplier 2: pass to PEs on the right in one row
		stream<matrix_t> &result)//final results of one column
{
	matrix_t A_temp[matrix_size];
	matrix_t L_temp[matrix_size];

	for(ap_uint<iterator_bit> i = 0; i < matrix_size; i++)//read in the operands of the first column
	{
#pragma HLS pipeline II=1
		A_temp[i] = in_A.read();
	}
	for(int i = 0; i < (matrix_size - 1)*(matrix_size)/2; i++)//pass the submatrix to the next PE
	{
#pragma HLS pipeline II=1
		out_A.write(in_A.read());
	}

	L_temp[0] = x_sqrt(A_temp[0]);
	result.write(L_temp[0]);

	for(ap_uint<iterator_bit> i = 1; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		L_temp[i] = A_temp[i] / L_temp[0];//calculate each element in the column
		result.write(L_temp[i]);
		if(i == 1)//get this logic by induction
		{
			out_row.write(L_temp[i]);
			out_col.write(L_temp[i]);
		}
		else
			out_row.write(L_temp[i]);
	}
}

//id starts from 1
void PE(ap_int<iterator_bit> id,
	stream<matrix_t> &in_A,
	stream<matrix_t> &out_A,//pass operand A to the next PE
	stream<matrix_t> &in_col,
	stream<matrix_t> &in_row,
	stream<matrix_t> &out_col,//multiplier 1: shared in the same column
	stream<matrix_t> &out_row,//multiplier 2: pass to PEs on the right in one row
	stream<matrix_t> &result)//final results of one column
{
	matrix_t A_temp[matrix_size];
	matrix_t L_temp[matrix_size];
	matrix_t col_L;
	matrix_t row_L;

//read in and pass out operands in matrix A
	loop_Aread: for(ap_uint<iterator_bit> i = id; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		A_temp[i] = in_A.read();
	}
	loop_Awrite: for(int i = 0; i < (matrix_size - id)*(matrix_size - id - 1)/2; i++)
	{
#pragma HLS pipeline II=1
		out_A.write(in_A.read());
	}

//update elements in A with input L(i,j) from 'in_col' and 'in_row'
	loop_mps: for(ap_uint<iterator_bit> i = 0; i < id; i++)
	{
		col_L = in_col.read();
		for(ap_uint<iterator_bit> j = id; j < matrix_size; j++)
		{
#pragma HLS pipeline II=1
			row_L = in_row.read();
			A_temp[j] = A_temp[j] - row_L * col_L;
			if(j == id + 1)
			{
				out_col.write(row_L);
				out_row.write(row_L);
			}
			else if(j > id + 1)
				out_row.write(row_L);
		}
	}

//calculate final results of L and pass out respectively to 'out_col' and 'out_row'
	A_temp[id] = x_sqrt(A_temp[id]);
	L_temp[id] = A_temp[id];
	loop_div: for(ap_uint<iterator_bit> i = id + 1; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		L_temp[i] = A_temp[i] / A_temp[id];
		if(i == id + 1)
		{
			out_col.write(L_temp[i]);
			out_row.write(L_temp[i]);
		}
		else
			out_row.write(L_temp[i]);
	}

//output final results to the collector
	loop_write: for(ap_uint<iterator_bit> i = id; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		result.write(L_temp[i]);
	}
}

//PE_secondlast differs from PE in that it does not contain FIFO 'out_row'
void PE_secondlast(ap_int<iterator_bit> id,
		stream<matrix_t> &in_A,
		stream<matrix_t> &out_A,
		stream<matrix_t> &in_col,
		stream<matrix_t> &in_row,
		stream<matrix_t> &out_col,
		stream<matrix_t> &result)
{
	matrix_t A_temp[matrix_size];
	matrix_t col_L;
	matrix_t row_L;

//read in and pass out operands in matrix A
	loop_Aread: for(ap_uint<iterator_bit> i = id; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		A_temp[i] = in_A.read();
	}

	out_A.write(in_A.read());

//update elements in A with input L(i,j) from 'in_col' and 'in_row'
	loop_mps: for(ap_uint<iterator_bit> i = 0; i < id; i++)
	{

		col_L = in_col.read();
		for(ap_uint<iterator_bit> j = id; j < matrix_size; j++)
		{
#pragma HLS pipeline II=1
			row_L = in_row.read();
			A_temp[j] = A_temp[j] - row_L * col_L;
			if(j == id + 1)
			{
				out_col.write(row_L);
			}
		}
	}

//calculate final results of L and pass out to 'out_col'
	A_temp[id] = x_sqrt(A_temp[id]);
	result.write(A_temp[id]);

	A_temp[id + 1] = A_temp[id + 1] / A_temp[id];
	out_col.write(A_temp[id + 1]);
	result.write(A_temp[id + 1]);
}

//PE_last preforms subtraction of products and a final square-root 
void PE_last(stream<matrix_t> &in_A,
		stream<matrix_t> &in_col,
		stream<matrix_t> &result)
{
	matrix_t A_temp;
	matrix_t col_L;

	A_temp = in_A.read();
	for(ap_uint<iterator_bit> i = 1; i < matrix_size; i++)
	{
		col_L = in_col.read();
		A_temp = A_temp - col_L * col_L;
	}

	A_temp = x_sqrt(A_temp);
	result.write(A_temp);
}

//collector: collects final results in column major order
void collector(
		stream<matrix_t> &feedout0,
		stream<matrix_t> &feedout1,
		stream<matrix_t> &feedout2,
		stream<matrix_t> &feedout3,
		matrix_t L[matrix_size][matrix_size])
{
	for(ap_uint<iterator_bit> i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		L[i][0] = feedout0.read();
	}

	for(ap_uint<iterator_bit> i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		if(i >= 1)
			L[i][1] = feedout1.read();
		else
			L[i][1] = 0;
	}

	for(ap_uint<iterator_bit> i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		if(i >= 2)
			L[i][2] = feedout2.read();
		else
			L[i][2] = 0;
	}

	for(ap_uint<iterator_bit> i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		if(i >= 3)
			L[i][3] = feedout3.read();
		else
			L[i][3] = 0;
	}

}

int top(matrix_t A[matrix_size][matrix_size],
	matrix_t L[matrix_size][matrix_size])
{
#pragma HLS dataflow
	stream<matrix_t> feedin;
#pragma HLS stream variable=feedin depth=3

	stream<matrix_t> pass_A1;
#pragma HLS stream variable=pass_A1 depth=5
	stream<matrix_t> pass_A2;
#pragma HLS stream variable=pass_A2 depth=5
	stream<matrix_t> pass_A3;
#pragma HLS stream variable=pass_A3 depth=5

	stream<matrix_t> pass_col1;
#pragma HLS stream variable=pass_col1 depth=5
	stream<matrix_t> pass_col2;
#pragma HLS stream variable=pass_col2 depth=5
	stream<matrix_t> pass_col3;
#pragma HLS stream variable=pass_col3 depth=5

	stream<matrix_t> pass_row1;
#pragma HLS stream variable=pass_row1 depth=5
	stream<matrix_t> pass_row2;
#pragma HLS stream variable=pass_row2 depth=5

	stream<matrix_t> feedout0;
#pragma HLS stream variable=feedout0 depth=5
	stream<matrix_t> feedout1;
#pragma HLS stream variable=feedout1 depth=5
	stream<matrix_t> feedout2;
#pragma HLS stream variable=feedout2 depth=5
	stream<matrix_t> feedout3;
#pragma HLS stream variable=feedout3 depth=5

	feeder(A, feedin);
	PE_head(feedin, pass_A1, pass_col1, pass_row1, feedout0);
	PE(1, pass_A1, pass_A2, pass_col1, pass_row1, pass_col2, pass_row2, feedout1);
	PE_secondlast(2, pass_A2, pass_A3, pass_col2, pass_row2, pass_col3, feedout2);
	PE_last(pass_A3, pass_col3, feedout3);

	collector(
		feedout0,
		feedout1,
		feedout2,
		feedout3,
	       	L);
	return 0;
}

