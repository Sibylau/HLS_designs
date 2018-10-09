//; $ROW = param_define("ROW", 4);
//; $COL = param_define("COL", 3);
//; $BIT_R = param_define("BIT_R", 3);
//; $BIT_C = param_define("BIT_C", 3);
//This program computes QR decomposition of A
//using Givens Rotations with parallel rotation generation.

#include "qr.h"
#include <iostream>

using namespace std;

//; $COL_1 = $COL - 1;
void feeder(MATRIX_T A[ROWS][COLS],
//; for($i = 0; $i < $COL - 1; $i++)
//; {
	stream<MATRIX_T> &feedin`$i`,
//; }
	stream<MATRIX_T> &feedin`$COL_1`)
{
#pragma HLS array_partition variable=A complete dim=2
//; for($i = 0; $i < $COL; $i++)
//; {
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		feedin`$i`.write(A[i][`$i`]);
	}
//; }
}

MATRIX_T qrf_mag(MATRIX_T a, MATRIX_T b)
{
#pragma HLS inline
	MATRIX_T aa = a * a;
	MATRIX_T bb = b * b;
	MATRIX_T mag = x_sqrt(aa + bb);
	return mag; 
}

/*void qrf_givens(MATRIX_T x, MATRIX_T y, MATRIX_T &c, MATRIX_T &s, MATRIX_T &r)
{
	r = qrf_mag(x, y);

	c = x / r;
	s = y / r;
}*/

//can be used for both left mm and right mm
//for left mm: 
// [a b]|c -s|
//		|s  c|
//for right mm:
// | c s||a|
// |-s c||b|
void qrf_mm(MATRIX_T c, MATRIX_T s, MATRIX_T &op1, MATRIX_T &op2)
{
#pragma HLS inline
	MATRIX_T a = op2 * s + op1 * c;
	MATRIX_T b = op2 * c - op1 * s;

	op1 = a;
	op2 = b;
}

//if (ROWS > COLS) 
//	the last PE1 is needed;
//if (ROWS == COLS)
//  the last PE1 will be omitted;
void PE1(uint_i id,
		stream<MATRIX_T> &in,
		stream<MATRIX_T> &pass_c,
		stream<MATRIX_T> &pass_s,
		stream<MATRIX_T> &out_c,
		stream<MATRIX_T> &out_s,
		stream<MATRIX_T> &out_R)
{
	MATRIX_T A[ROWS];

	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		A[i] = in.read();
	}

	for(uint_i i = ROWS - 1; i > id; i--)
	{
#pragma HLS loop_flatten off
		if(A[i] == 0)
		{
			MATRIX_T c = 1;
			MATRIX_T s = 0;
			pass_c.write(c);
			out_c.write(c);
			pass_s.write(s);
			out_s.write(s);
		}
		else
		{
			MATRIX_T mag = qrf_mag(A[i], A[i - 1]);
			MATRIX_T c = A[i - 1] / mag;
			MATRIX_T s = A[i] / mag;
			A[i - 1] = mag;
			A[i] = 0;
			pass_c.write(c);
			out_c.write(c);
			pass_s.write(s);
			out_s.write(s);
		}
	}
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		out_R.write(A[i]);
	}
}

void PE1_tail(
		stream<MATRIX_T> &in,
		stream<MATRIX_T> &out_c,
		stream<MATRIX_T> &out_s,
		stream<MATRIX_T> &out_R)
{
	MATRIX_T A[ROWS];

	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		A[i] = in.read();
	}
	for(uint_i i = ROWS - 1; i >= COLS; i--)
	{
#pragma HLS loop_flatten off
		if(A[i] == 0)
		{
			MATRIX_T c = 1;
			MATRIX_T s = 0;
			out_c.write(c);
			out_s.write(s);
		}
		else
		{
			MATRIX_T mag = qrf_mag(A[i], A[i - 1]);
			MATRIX_T c = A[i - 1] / mag;
			MATRIX_T s = A[i] / mag;
			A[i - 1] = mag;
			A[i] = 0;
			out_c.write(c);
			out_s.write(s);
		}
	}
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		out_R.write(A[i]);
	}
}

void PE2(uint_i idx,
		uint_i idy,
		stream<MATRIX_T> &in,
		stream<MATRIX_T> &in_c,
		stream<MATRIX_T> &in_s,
		stream<MATRIX_T> &out_c,
		stream<MATRIX_T> &out_s,
		stream<MATRIX_T> &out_mid)
{
	MATRIX_T A[ROWS];

	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		A[i] = in.read();
	}

	for(uint_i i = ROWS - 1; i > idx; i--)
	{
		MATRIX_T c = in_c.read();
		MATRIX_T s = in_s.read();
		qrf_mm(c, s, A[i - 1], A[i]);
		out_c.write(c);
		out_s.write(s);
	}
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		out_mid.write(A[i]);
	}
}

void PE2_tail(
		uint_i idx,
		stream<MATRIX_T> &in,
		stream<MATRIX_T> &in_c,
		stream<MATRIX_T> &in_s,
		stream<MATRIX_T> &out_mid)
{
	MATRIX_T A[ROWS];

	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		A[i] = in.read();
	}
	for(uint_i i = ROWS - 1; i > idx; i--)
	{
		MATRIX_T c = in_c.read();
		MATRIX_T s = in_s.read();
		qrf_mm(c, s, A[i - 1], A[i]);
	}
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		out_mid.write(A[i]);
	}
}

void collector(
//; for($i = 0; $i < $COL; $i++)
//; {
		stream<MATRIX_T> &in_c`$i`,
//; }
//; for($i = 0; $i < $COL; $i++)
//; {
		stream<MATRIX_T> &in_s`$i`,
//; }
//; for($i = 0; $i < $COL; $i++)
//; {
		stream<MATRIX_T> &in_R`$i`,
//; }
		MATRIX_T Q[ROWS][ROWS],
		MATRIX_T R[ROWS][COLS])
{
	//initialize for Q
	MATRIX_T Q_i[ROWS][ROWS];

#pragma HLS array_partition variable=Q_i cyclic factor=2 dim=2
	for(uint_i i = 0; i < ROWS; i++)
	{
		#pragma HLS loop_merge force
		for (uint_i j = 0; j < ROWS; j++)
		{
			#pragma HLS pipeline II=1
			if(i == j)
			{
				Q_i[i][j] = 1;
			}
			else
			{
				Q_i[i][j] = 0;
			}
		}
		for(uint_i j = 0; j < COLS; j++)
		{
			#pragma HLS pipeline II=1
			R[i][j] = 0;
		}
	}

//; for($i = 0; $i < $COL; $i++)
//; {
	for(uint_i i = ROWS - 1; i > `$i`; i--)
	{
		MATRIX_T c = in_c`$i`.read();
		MATRIX_T s = in_s`$i`.read();
		for(uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			qrf_mm(c, s, Q_i[j][i - 1], Q_i[j][i]);
		}
	}
	for(uint_i i = 0; i < ROWS; i++)
	{
#pragma HLS pipeline II=1
		R[i][`$i`] = in_R`$i`.read();
	}

//; }
	for(uint_i i = 0; i < ROWS; i++)
	{
		for (uint_i j = 0; j < ROWS; j++)
		{
#pragma HLS pipeline II=1
			Q[i][j] = Q_i[i][j];
		}
	}
}

int top(MATRIX_T A[ROWS][COLS],
	MATRIX_T Q[ROWS][ROWS],
	MATRIX_T R[ROWS][COLS])
{
	//check QR operant
	if(ROWS < COLS)
	{
	#ifndef __SYNTHESIS__
        printf("ERROR: Parameter error - RowsA must be greater than ColsA; currently RowsA = %d ColsA = %d\n",ROWS,COLS);
	#endif
        exit(1);	
	}

	#pragma HLS dataflow

//; for($i = 0; $i < $COL; $i++)
//; {
	stream<MATRIX_T> feedin`$i`;
#pragma HLS stream variable=feedin`$i` depth=10
//; }

//; for($i = 0; $i < $COL_1; $i++)
//; {
//;  for($j = 0; $j < $COL_1 - $i; $j++)
//;  {
	stream<MATRIX_T> pass_c`$i`_`$j`;
#pragma HLS stream variable=pass_c`$i`_`$j` depth=10
//;  }
//; }

//; for($i = 0; $i < $COL_1; $i++)
//; {
//;  for($j = 0; $j < $COL_1 - $i; $j++)
//;  {
	stream<MATRIX_T> pass_s`$i`_`$j`;
#pragma HLS stream variable=pass_s`$i`_`$j` depth=10
//;  }
//; }

//; for($i = 0; $i < $COL; $i++)
//; {
	stream<MATRIX_T> out_c`$i`;
#pragma HLS stream variable=out_c`$i` depth=10
//; }

//; for($i = 0; $i < $COL; $i++)
//; {
	stream<MATRIX_T> out_s`$i`;
#pragma HLS stream variable=out_s`$i` depth=10
//; }

//; for($i = 0; $i < $COL; $i++)
//; {
	stream<MATRIX_T> out_R`$i`;
#pragma HLS stream variable=out_R`$i` depth=10
//; }

//; for($i = 0; $i < $COL_1; $i++)
//; {
//;  for($j = $i + 1; $j < $COL; $j++)
//;  {
	stream<MATRIX_T> mid`$i`_`$j`;
#pragma HLS stream variable=mid`$i`_`$j` depth=10
//;  }
//; }

	feeder(A, 
//; for($i = 0; $i < $COL_1; $i++)
//; {
		feedin`$i`,
//; }
		feedin`$COL_1`);

	PE1(0, feedin0, pass_c0_0, pass_s0_0, out_c0, out_s0, out_R0);
//; $index = 1;
//; for($index = 1; $index < $COL_1; $index++)
//; {
//; $i_1 = $index - 1;
	PE2(0, `$index`, feedin`$index`, pass_c0_`$i_1`, pass_s0_`$i_1`, pass_c0_`$index`, pass_s0_`$index`, mid0_`$index`);
//; }
//; $i_1 = $index - 1;
	PE2_tail(0, feedin`$COL_1`, pass_c0_`$i_1`, pass_s0_`$i_1`, mid0_`$index`);

//; for($i = 1; $i < $COL_1; $i++)
//; {
//; $pass = 0;
//; $i_m1 = $i - 1;
//; $mid = $i_m1 + 1;
	PE1(`$i`, mid`$i_m1`_`$mid`, pass_c`$i`_`$pass`, pass_s`$i`_`$pass`, out_c`$i`, out_s`$i`, out_R`$i`);
//;  for($j = 1; $j < $COL_1 - $i; $j++)
//;  {
//;  $mid = $mid + 1;
//;  $pass = $pass + 1;
//;  $pass_1 = $pass - 1;
	PE2(`$i`, `$j`, mid`$i_m1`_`$mid`, pass_c`$i`_`$pass_1`, pass_s`$i`_`$pass_1`, pass_c`$i`_`$pass`, pass_s`$i`_`$pass`, mid`$i`_`$mid`);
//;  }
//; $mid = $mid + 1;
//; if(($ROW == $COL) && ($i == $COL_1 - 1))
//; {
	PE2_tail(`$i`, mid`$i_m1`_`$mid`, pass_c`$i`_`$pass`, pass_s`$i`_`$pass`, out_R`$COL_1`);
//; }
//; else
//; {
	PE2_tail(`$i`, mid`$i_m1`_`$mid`, pass_c`$i`_`$pass`, pass_s`$i`_`$pass`, mid`$i`_`$mid`);
//; }
//; }

//; if($ROW > $COL)
//; {
//; $mid_1st = $COL - 2;
//; $mid_2nd = $mid_1st + 1;
	PE1_tail(mid`$mid_1st`_`$mid_2nd`, out_c`$COL_1`, out_s`$COL_1`, out_R`$COL_1`);
//; }

	collector(
//; for($i = 0; $i < $COL; $i++)
//; {
		out_c`$i`,
//; }
//; for($i = 0; $i < $COL; $i++)
//; {
	       	out_s`$i`,
//; }
//; for($i = 0; $i < $COL; $i++)
//; {
	       	out_R`$i`,
//; }
		Q, R);
	return 0;
}
