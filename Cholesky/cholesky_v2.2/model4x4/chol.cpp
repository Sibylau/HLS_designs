//This is a 1D systolic array implementation of the revised right-looking Cholesky algorithm.
#include "chol.h"
#include <iostream>

using namespace std;

//Feeder: feed in the elements in the lower triangular matrix of A
//   	  (Note that A is a symmetric matrix) in column major order.
void feeder(matrix_t A[matrix_size][matrix_size],
	stream<matrix_t> &feedin,
	stream<matrix_t> &A_pass)
{
	ap_uint<iterator_bit> i, j;

	A_pass.write(A[0][0]);//pass A(0,0) successively to PE_tail
	for(i = 0; i < matrix_size; i++)
	{
		for(j = 0; j < matrix_size; j++)
		{
#pragma HLS pipeline II=1
			if(j >= i)
				feedin.write(A[j][i]);
		}
	}
}
//PE_head differs from PE in no existing of 'mid_in' and 'M_passin' fifos
void PE_head(stream<matrix_t> &feedin,
	stream<matrix_t> &A_passin,
	stream<matrix_t> &A_passout,//pass A* successively to PE_tail
	stream<matrix_t> &M_passout,//pass the generated variable M to the next PE
	stream<matrix_t> &mid_out)//pass the updated middle results to the next PE
{
	ap_uint<iterator_bit> i ,j;
	matrix_t mid_L[matrix_size][matrix_size];
	matrix_t mid_M[matrix_size];
	matrix_t diag;
	matrix_t temp_a;

	A_passout.write(A_passin.read());//pass directly A(0,0) to PE_tail

	diag = feedin.read();//to cancel the dependency and achieve II=1
	for(i = 1; i < matrix_size; i++)//loop to generate variable M
	{
#pragma HLS pipeline II=1
		mid_L[i][0] = feedin.read();
		mid_M[i] = mid_L[i][0] / diag;
		M_passout.write(mid_M[i]);//output variable M
	}

	for(i = 1; i < matrix_size; i++)//loop to update submatrix of A
	{
		for(j = 1 ; j < matrix_size; j++)
		{
#pragma HLS pipeline II=1
			if(j >= i)
			{
			temp_a = mid_L[j][0];
			mid_L[j][i] = feedin.read();
			mid_L[j][i] = mid_L[j][i] - temp_a * mid_M[i];
			mid_out.write(mid_L[j][i]);
			}
		}
	}

	A_passout.write(mid_L[1][1]);//PE_head is special here: pass both A(0,0) and A(1,1)
}

//id = 1, from 2nd PE
void PE(ap_uint<iterator_bit> id,
	stream<matrix_t> &A_passin,
	stream<matrix_t> &A_passout,//pass A* successively to PE_tail
	stream<matrix_t> &M_passin,
	stream<matrix_t> &M_passout,//pass the generated variable M to the next PE
	stream<matrix_t> &mid_in,
	stream<matrix_t> &mid_out)//pass the updated middle results to the next PE
{
	int i, j;
	
	matrix_t mid_M[matrix_size];
	matrix_t mid_L[matrix_size][matrix_size];
	matrix_t diag;
	matrix_t temp_a;
	matrix_t temp_read[matrix_size];

	int bound = (max_length) - ((matrix_size - id) * (matrix_size - id - 1) / 2);
	for(i = 0; i < bound; i++)//equal to 'while(!empty)', but more reliable
	{
#pragma HLS pipeline II=1
		M_passout.write(M_passin.read());
	}

	diag = mid_in.read();
	for(i = id + 1; i < matrix_size; i++)
	{
#pragma HLS pipeline II=1
		mid_L[i][id] = mid_in.read();
		mid_M[i] = mid_L[i][id] / diag;
		M_passout.write(mid_M[i]);
	}

	for(i = id + 1; i < matrix_size; i++)
	{
		for(j = id + 1 ; j < matrix_size; j++)
		{
#pragma HLS pipeline II=1
			if(j >= i)
			{
				temp_a = mid_L[j][id];
				mid_L[j][i] = mid_in.read();
				mid_L[j][i] = mid_L[j][i] - temp_a * mid_M[i];
				mid_out.write(mid_L[j][i]);
			}
		}
	}

	for(i = 0; i <= id; i++)//loop to inherit A* from its predecessor
	{
#pragma HLS pipeline II=1
		temp_read[i] = A_passin.read();
		A_passout.write(temp_read[i]);
	}
	A_passout.write(mid_L[id + 1][id + 1]);//only add one diagonal element to it
}

//PE_last differs in no existing of 'mid_out'
void PE_last(stream<matrix_t> &A_passin,
	stream<matrix_t> &A_passout,
	stream<matrix_t> &M_passin,
	stream<matrix_t> &M_passout,
	stream<matrix_t> &mid_in)
{
	int i;
	matrix_t mid_L[matrix_size][matrix_size];
	matrix_t mid_M;
	matrix_t diag;
	matrix_t temp_a;
	matrix_t temp_read[matrix_size];

	int bound = max_length - 1;
	for(i = 0; i < bound; i++)
	{
#pragma HLS pipeline II=1
		M_passout.write(M_passin.read());
	}

//the single iteration from normal PE
	diag = mid_in.read();
	mid_L[matrix_size - 1][matrix_size - 2] = mid_in.read();
	mid_M = mid_L[matrix_size - 1][matrix_size - 2] / diag;
	M_passout.write(mid_M);

	temp_a = mid_L[matrix_size - 1][matrix_size - 2];
	mid_L[matrix_size - 1][matrix_size - 1] = mid_in.read();
	mid_L[matrix_size - 1][matrix_size - 1] = mid_L[matrix_size - 1][matrix_size - 1] - temp_a * mid_M;

	for(i = 0; i < matrix_size - 1; i++)
	{
#pragma HLS pipeline II=1
		temp_read[i] = A_passin.read();
		A_passout.write(temp_read[i]);
	}
	A_passout.write(mid_L[matrix_size - 1][matrix_size - 1]);
}

//PE_tail is special
void PE_tail(stream<matrix_t> &A_in,
	stream<matrix_t> &M_in,
	stream<matrix_t> &feedout)
{
	ap_uint<iterator_bit> i, j;
	matrix_t L_temp[matrix_size];
	matrix_t M_temp[matrix_size][matrix_size];
	matrix_t result;

	storem_loop:for(i = 0; i < matrix_size - 1; i++)//to store M in M_temp
	for(j = 1; j < matrix_size; j++)
	{
#pragma HLS pipeline II=1
		if(j > i)
			M_temp[j][i] = M_in.read();
	}

	sqrt_loop: for(i = 0; i < matrix_size; i++)//to realize pipeline of 4 sqrt operations
	{
#pragma HLS pipeline II=1
		L_temp[i] = x_sqrt(A_in.read());
		feedout.write(L_temp[i]);
	}

	mul_loop: for(i = 0; i < matrix_size - 1; i++)//to obtain the final results by multiplying with M 
	{
		for(j = 1; j < matrix_size; j++)
		{
#pragma HLS pipeline II=1

			if(j > i)
			{
				result = M_temp[j][i] * L_temp[i];
				feedout.write(result);
			}
		}
	}
}

void collector(stream<matrix_t> &feedout,
	matrix_t L[matrix_size][matrix_size])
{
	ap_uint<iterator_bit> i, j;

	for(i = 0; i < matrix_size; i++)//initialize L with zeros
	{
		for(j = 0; j < matrix_size; j++)
		{
#pragma HLS pipeline II=1
			L[j][i] = 0;
		}
	}

	for(i = 0; i < matrix_size; i++)//read in diagonal elements
	{
#pragma HLS pipeline II=1
		L[i][i] = feedout.read();
	}

	for(i = 0; i < matrix_size; i++)//read in non-diagonal elements
	{
		for(j = 0; j < matrix_size; j++)
		{
#pragma HLS pipeline II=1
			if(j > i)
				L[j][i] = feedout.read();
		}
	}
}


int top(matrix_t A[matrix_size][matrix_size],
	matrix_t L[matrix_size][matrix_size])
{
#pragma HLS DATAFLOW

	stream<matrix_t> feedin, feedout;
#pragma HLS STREAM variable=feedin depth=5
#pragma HLS STREAM variable=feedout depth=10//to avoid potential fifo stalls

	stream<matrix_t> A_pass1;
#pragma HLS STREAM variable=A_pass1 depth=10
	stream<matrix_t> A_pass2;
#pragma HLS STREAM variable=A_pass2 depth=10
	stream<matrix_t> A_pass3;
#pragma HLS STREAM variable=A_pass3 depth=10
	stream<matrix_t> A_pass4;
#pragma HLS STREAM variable=A_pass4 depth=10

	stream<matrix_t> M_passout0;
#pragma HLS STREAM variable=M_passout0 depth=10
	stream<matrix_t> M_passout1;
#pragma HLS STREAM variable=M_passout1 depth=10
	stream<matrix_t> M_passout2;
#pragma HLS STREAM variable=M_passout2 depth=10

	stream<matrix_t> mid0;
#pragma HLS STREAM variable=mid0 depth=10
	stream<matrix_t> mid1;
#pragma HLS STREAM variable=mid1 depth=10

	feeder(A, feedin, A_pass1);

	PE_head(feedin, A_pass1, A_pass2, M_passout0, mid0);

	PE(1, A_pass2, A_pass3, M_passout0, M_passout1, mid0, mid1);
	PE_last(A_pass3, A_pass4, M_passout1, M_passout2, mid1);

	PE_tail(A_pass4, M_passout2, feedout);

	collector(feedout, L);
	return 0;

}
