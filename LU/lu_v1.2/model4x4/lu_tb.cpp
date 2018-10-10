#include "lu.h"
#include "hls/linear_algebra/utils/x_hls_matrix_utils.h"
//#include "hls/linear_algebra/utils/x_hls_matrix_tb_utils.h"
#include <iostream>
using namespace std;

int main ()
{
	int lu_success = 0;
	matrix_t A[matrix_size][matrix_size];
	matrix_t A1[matrix_size][matrix_size];
	matrix_t A2[matrix_size][matrix_size];
	matrix_t L[matrix_size][matrix_size];
	matrix_t U[matrix_size][matrix_size];
	uint_i P[matrix_size];
	matrix_t L1[matrix_size][matrix_size];
	matrix_t U1[matrix_size][matrix_size];
	uint_i P1[matrix_size];
	matrix_t L2[matrix_size][matrix_size];
	matrix_t U2[matrix_size][matrix_size];
	uint_i P2[matrix_size];

	FILE *fp;
	fp = fopen("/curr/jieliu15/HLS_design/LU/op4.bin","rb");
	for(int i = 0; i < matrix_size; i++)
	{
		fread(A[i], sizeof(float), matrix_size, fp);
	}
	fclose(fp);

	for(int i = 0; i < matrix_size; i++)
	{
		for(int j = 0; j < matrix_size; j++)
		{
			A1[i][j] = A[i][j];
			A2[i][j] = A[i][j];
		}
	}
	//1 : SA
	for(int i = 0; i < 10; i++)
		lu_success = top(A, L, U, P);
	
	for(int i = 0; i < matrix_size; i++)
	{
		P1[i] = 0;
		P2[i] = 0;
		for(int j = 0; j < matrix_size; j++)
		{
			U1[i][j] = 0;
			L2[i][j] = 0;
		}
	}
	for(int i = 0; i < matrix_size; i++)
	{
		for(int j = 0; j < matrix_size; j++)
		{
			if(i == j)
			{
				L1[i][j] = 1;
				U2[i][j] = 1;
			}
			else
			{
				L1[i][j] = 0;
				U2[i][j] = 0;
			}
		}
	}

	// 2 : Doolittle
	for(int i = 0; i < matrix_size - 1; i++)
	{
		matrix_t maxpwr = hls::abs(A1[i][i]);
		P1[i] = i;
		for(int j = i + 1; j < matrix_size; j++)
		{
			if(maxpwr < hls::abs(A1[j][i]))
			{
				maxpwr = hls::abs(A1[j][i]);
				P1[i] = j;
			}
		}
		if(P1[i] != i)
		{
			uint_i index = P1[i];
			for(int j = i; j < matrix_size; j++)
			{
				maxpwr = A1[index][j];
				A1[index][j] = A1[i][j];
				A1[i][j] = maxpwr;
			}
		}
		for(int j = i; j < matrix_size; j++)
			U1[i][j] = A1[i][j];
		for(int j = i + 1; j < matrix_size; j++)
			L1[j][i] = A1[j][i] / A1[i][i];
		for(int j = i + 1; j < matrix_size; j++)
		{
			for(int k = i + 1; k < matrix_size; k++)
			{
				A1[k][j] = A1[k][j] - L1[k][i] * U1[i][j];
			}
		}
	}
	P1[matrix_size - 1] = matrix_size - 1;
	U1[matrix_size - 1][matrix_size - 1] = A1[matrix_size - 1][matrix_size - 1];

	// 3 : Crout
	for(int i = 0; i < matrix_size - 1; i++)
	{
		matrix_t maxpwr = hls::abs(A2[i][i]);
		P2[i] = i;
		for(int j = i + 1; j < matrix_size; j++)
		{
			if(maxpwr < hls::abs(A2[j][i]))
			{
				maxpwr = hls::abs(A2[j][i]);
				P2[i] = j;
			}
		}
		if(P2[i] != i)
		{
			uint_i index = P2[i];
			for(int j = i; j < matrix_size; j++)
			{
				maxpwr = A2[index][j];
				A2[index][j] = A2[i][j];
				A2[i][j] = maxpwr;
			}
		}
		for(int j = i; j < matrix_size; j++)
			L2[j][i] = A2[j][i];
		for(int j = i + 1; j < matrix_size; j++)
			U2[i][j] = A2[i][j] / A2[i][i];
		for(int j = i + 1; j < matrix_size; j++)
		{
			for(int k = i + 1; k < matrix_size; k++)
			{
				A2[k][j] = A2[k][j] - L2[k][i] * U2[i][j];
			}
		}
	}
	P2[matrix_size - 1] = matrix_size - 1;
	L2[matrix_size - 1][matrix_size - 1] = A2[matrix_size - 1][matrix_size - 1];

    printf("A = \n");
    hls::print_matrix<matrix_size, matrix_size, matrix_t, hls::NoTranspose>(A, "   ");

    printf("SA:\n");
    printf("L = \n");
    hls::print_matrix<matrix_size, matrix_size, matrix_t, hls::NoTranspose>(L, "   ");
    printf("U = \n");
    hls::print_matrix<matrix_size, matrix_size, matrix_t, hls::NoTranspose>(U, "   ");
    printf("P = \n");
    for(int i = 0; i < matrix_size; i++)
    {
    	cout<<P[i]<<" ";
    }
    printf("\n");

    printf("Doolittle:\n");
    printf("L1 = \n");
    hls::print_matrix<matrix_size, matrix_size, matrix_t, hls::NoTranspose>(L1, "   ");
    printf("U1 = \n");
    hls::print_matrix<matrix_size, matrix_size, matrix_t, hls::NoTranspose>(U1, "   ");
    printf("P1 = \n");
    for(int i = 0; i < matrix_size; i++)
        {
    	cout<<P1[i]<<" ";
        }
        printf("\n");

	printf("Crout:\n");
    printf("L2 = \n");
    hls::print_matrix<matrix_size, matrix_size, matrix_t, hls::NoTranspose>(L2, "   ");
    printf("U2 = \n");
    hls::print_matrix<matrix_size, matrix_size, matrix_t, hls::NoTranspose>(U2, "   ");
    printf("P2 = \n");
    for(int i = 0; i < matrix_size; i++)
        {
        	cout<<P2[i]<<" ";
        }
        printf("\n");

    // Generate error ratio and compare against threshold value
    // - LAPACK error measurement method
    // - Threshold taken from LAPACK test functions
    int fail = 0;
    int i, j;
    float error = 0;
    for(i = 0; i < matrix_size; i++)
    	for(j = 0; j < matrix_size; j++)
    	{
    		error += hls::abs(L[i][j]- L1[i][j]);
    		error += hls::abs(U[i][j]- U1[i][j]);
    	}
    if(error > 5.0)
    	fail = 1;
    cout<<"error: "<<error<<endl;
    return (lu_success||fail);
}
