//; $SIZE = param_define("SIZE", 4);
#include "chol.h"
#include <iostream>

using namespace std;

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

void PE_head(stream<matrix_t> &in_A,
		stream<matrix_t> &out_A,
		stream<matrix_t> &out_col,
		stream<matrix_t> &out_row,
		stream<matrix_t> &result)
{
	matrix_t A_temp[matrix_size];
	matrix_t L_temp[matrix_size];

	for(ap_uint<iterator_bit> i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		A_temp[i] = in_A.read();
	}
	for(int i = 0; i < (matrix_size - 1)*(matrix_size)/2; i++)
	{
#pragma HLS pipeline II=1
		out_A.write(in_A.read());
	}

	L_temp[0] = x_sqrt(A_temp[0]);
	result.write(L_temp[0]);

	for(ap_uint<iterator_bit> i = 1; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		L_temp[i] = A_temp[i] / L_temp[0];
		result.write(L_temp[i]);
		if(i == 1)
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
	stream<matrix_t> &out_A,
	stream<matrix_t> &in_col,
	stream<matrix_t> &in_row,
	stream<matrix_t> &out_col,
	stream<matrix_t> &out_row,
	stream<matrix_t> &result)
{
	matrix_t A_temp[matrix_size];
	matrix_t L_temp[matrix_size];
	matrix_t col_L;
	matrix_t row_L;

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
	loop_write: for(ap_uint<iterator_bit> i = id; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		result.write(L_temp[i]);
	}
}

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

	loop_Aread: for(ap_uint<iterator_bit> i = id; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		A_temp[i] = in_A.read();
	}

	out_A.write(in_A.read());

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

	A_temp[id] = x_sqrt(A_temp[id]);
	result.write(A_temp[id]);

	A_temp[id + 1] = A_temp[id + 1] / A_temp[id];
	out_col.write(A_temp[id + 1]);
	result.write(A_temp[id + 1]);
}

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

void collector(
//; for($i = 0; $i < $SIZE; $i++)
//; {
		stream<matrix_t> &feedout`$i`,
//; }
		matrix_t L[matrix_size][matrix_size])
{
	for(ap_uint<iterator_bit> i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		L[i][0] = feedout0.read();
	}

//; for($i = 1; $i < $SIZE; $i++)
//; {
	for(ap_uint<iterator_bit> i = 0; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		if(i >= `$i`)
			L[i][`$i`] = feedout`$i`.read();
		else
			L[i][`$i`] = 0;
	}

//; }
}

int top(matrix_t A[matrix_size][matrix_size],
	matrix_t L[matrix_size][matrix_size])
{
#pragma HLS dataflow
	stream<matrix_t> feedin;
#pragma HLS stream variable=feedin depth=3

//; $SIZE_1 = $SIZE - 1;
//; for($i = 1; $i < $SIZE; $i++)
//; {
	stream<matrix_t> pass_A`$i`;
#pragma HLS stream variable=pass_A`$i` depth=5
//; }

//; for($i = 1; $i < $SIZE; $i++)
//; {
	stream<matrix_t> pass_col`$i`;
#pragma HLS stream variable=pass_col`$i` depth=5
//; }

//; for($i = 1; $i < $SIZE_1; $i++)
//; {
	stream<matrix_t> pass_row`$i`;
#pragma HLS stream variable=pass_row`$i` depth=5
//; }

//; for($i = 0; $i < $SIZE; $i++)
//; {
	stream<matrix_t> feedout`$i`;
#pragma HLS stream variable=feedout`$i` depth=5
//; }

	feeder(A, feedin);
	PE_head(feedin, pass_A1, pass_col1, pass_row1, feedout0);
//; $SIZE_2 = $SIZE - 2;
//; $A = 1;
//; $A_p1 = $A + 1;
//; $col = 1;
//; $col_p1 = $col + 1;
//; for($i = 1; $i < $SIZE_2; $i++)
//; {
	PE(`$i`, pass_A`$A`, pass_A`$A_p1`, pass_col`$col`, pass_row`$col`, pass_col`$col_p1`, pass_row`$col_p1`, feedout`$i`);
//; $A = $A + 1;
//; $A_p1 = $A + 1;
//; $col = $col + 1;
//; $col_p1 = $col + 1;
//; }
	PE_secondlast(`$A`, pass_A`$A`, pass_A`$A_p1`, pass_col`$col`, pass_row`$col`, pass_col`$col_p1`, feedout`$SIZE_2`);
	PE_last(pass_A`$A_p1`, pass_col`$col_p1`, feedout`$SIZE_1`);

	collector(
//; for($i = 0; $i < $SIZE; $i++)
//; {
		feedout`$i`,
//; }
	       	L);
	return 0;
}

