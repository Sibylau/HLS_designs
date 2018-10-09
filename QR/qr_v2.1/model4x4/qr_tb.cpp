#include "qr.h"
#include "hls/linear_algebra/utils/x_hls_matrix_utils.h"
//#include "hls/linear_algebra/utils/x_hls_MATRIX_Tb_utils.h"
using namespace std;

int main ()
{
	int qr_GR_success = 0;
	MATRIX_T A[ROWS][COLS];
	MATRIX_T A1[ROWS][COLS];
	MATRIX_T Q[ROWS][ROWS];
	MATRIX_T R[ROWS][COLS];
	MATRIX_T Q1[ROWS][ROWS];

	FILE *fp;
	fp = fopen("/curr/jieliu15/HLS_design/QR/op4x4.bin","rb");
	for(int i = 0; i < ROWS; i++)
	{
		fread(A[i], sizeof(float), COLS, fp);
	}
	fclose(fp);

	for(int i = 0; i < ROWS; i++)
	{
		for(int j = 0; j < COLS; j++)
		{
			A1[i][j] = A[i][j];
		}
	}
	//1 : SA
	for(int i = 0; i < 4; i++)
		qr_GR_success = top(A, Q, R);
	
	for(uint_i i = 0; i < ROWS; i++)
	{
		#pragma HLS loop_merge force
		for (uint_i j = 0; j < ROWS; j++)
		{
			if(i == j)
			{
				Q1[i][j] = 1;
			}
			else
			{
				Q1[i][j] = 0;
			}
		}
	}

	// 2 : GR
	for(int i = 0; i < COLS; i++)
	{
		for(int j = ROWS - 1; j > i; j--)
		{
			if(A1[j][i] == 0)
				continue;
			else
			{
				MATRIX_T mag = qrf_mag(A1[j][i], A1[j - 1][i]);
				MATRIX_T c = A1[j - 1][i] / mag;
				MATRIX_T s = A1[j][i] / mag;
				A1[j - 1][i] = mag;
				A1[j][i] = 0;
				for(int k = i + 1; k < COLS; k++)
				{
					qrf_mm(c, s, A1[j - 1][k], A1[j][k]);
				}
				for(int k = 0; k < ROWS; k++)
				{
					qrf_mm(c, s, Q1[k][j - 1], Q1[k][j]);
				}
			}
		}	
	}

    printf("A = \n");
    hls::print_matrix<ROWS, COLS, MATRIX_T, hls::NoTranspose>(A, "   ");

    printf("SA:\n");
    printf("Q = \n");
    hls::print_matrix<ROWS, ROWS, MATRIX_T, hls::NoTranspose>(Q, "   ");
    printf("R = \n");
    hls::print_matrix<ROWS, COLS, MATRIX_T, hls::NoTranspose>(R, "   ");

    printf("GR:\n");
    printf("Q1 = \n");
    hls::print_matrix<ROWS, ROWS, MATRIX_T, hls::NoTranspose>(Q1, "   ");
    printf("R1 = \n");
    hls::print_matrix<ROWS, COLS, MATRIX_T, hls::NoTranspose>(A1, "   ");

    // Generate error ratio and compare against threshold value
    // - LAPACK error measurement method
    // - Threshold taken from LAPACK test functions
    int fail = 0;
    int i, j;
    float error = 0;
    for(i = 0; i < ROWS; i++)
    {
    	for(j = 0; j < ROWS; j++)
    	{
    		error += hls::abs(Q[i][j]- Q1[i][j]);
    	}

    	for(j = 0; j < COLS; j++)
    	{
    		error += hls::abs(R[i][j]- A1[i][j]);
    	}
    }
    cout<<"error = "<<error<<endl;
    if(error > 5.0)
    	fail = 1;
    return (qr_GR_success||fail);
}

