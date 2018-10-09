/********* testbench of Cholesky **********/

#include "chol.h"
#include "hls/linear_algebra/utils/x_hls_matrix_utils.h"
//#include "hls/linear_algebra/utils/x_hls_matrix_tb_utils.h"
using namespace hls;

int main (void)
{
	int Cholesky_success = 0;
	matrix_t A[matrix_size][matrix_size];
	matrix_t A2[matrix_size][matrix_size];
	matrix_t L1[matrix_size][matrix_size];
	matrix_t L2[matrix_size][matrix_size];
	int i, j, k;
	float error = 0;

	FILE *fp;
	fp = fopen("/curr/jieliu15/HLS_design/Cholesky/op4.bin", "rb");
	for(int i = 0; i < matrix_size; i++)
	{
		fread(A[i], sizeof(float), matrix_size, fp);
	}
	fclose(fp);

	for(i = 0; i < matrix_size; i++)
	{
		for(j = 0; j < matrix_size; j++)
			A2[i][j] = A[i][j];
	}
	    for(i = 0; i < 5; i++)
	    {
	    	Cholesky_success |= top(A, L1);
	    }

	// Now re-create A: Ar = L * L'
    //hls::matrix_multiply<hls::NoTranspose,hls::Transpose,matrix_size,matrix_size,matrix_size,matrix_size,matrix_size,matrix_size,matrix_t,matrix_t>(L1, L1, Ar);

    for(i = 0; i < matrix_size; i++)
    {
    	L2[i][i] = x_sqrt(A2[i][i]);
    	for(j = i + 1; j < matrix_size; j++)
    	{
    		L2[j][i] = A2[j][i] / L2[i][i];
    		for(k = i + 1; k <= j; k++)
    			A2[j][k] = A2[j][k] - L2[j][i] * L2[k][i];
    	}
    }
    for(i = 0; i < matrix_size - 1; i++)
    	for(j = i + 1; j < matrix_size; j++)
    		L2[i][j] = 0;

    printf("A = \n");
    hls::print_matrix<matrix_size, matrix_size, matrix_t, hls::NoTranspose>(A, "   ");

   // printf("A2 = \n");
        //hls::print_matrix<matrix_size, matrix_size, matrix_t, hls::NoTranspose>(A2, "   ");

    printf("L1 = \n");
    hls::print_matrix<matrix_size, matrix_size, matrix_t, hls::NoTranspose>(L1, "   ");

    printf("L2 = \n");
    hls::print_matrix<matrix_size, matrix_size, matrix_t, hls::NoTranspose>(L2, "   ");

    // Generate error ratio and compare against threshold value
    // - LAPACK error measurement method
    // - Threshold taken from LAPACK test functions
    int fail = 0;
    for(i = 0; i < matrix_size; i++)
    {
    	for(j = 0; j < matrix_size; j++)
    		error += hls::abs(L1[i][j]- L2[i][j]);
    }

    printf("error = %.6f\n", error);
    if(error > 1e-5)
    	fail = 1;
    return (Cholesky_success||fail);
}
