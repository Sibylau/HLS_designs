# HLS_designs

Systolic array implementations for Cholesky, LU, and QR decomposition using HLS

## Get started
#### Environment
- Ubuntu 16.04.5 LTS
- Xilinx Vivado HLS v2017.4
- Matlab R2017a

#### Directory Tree

#### Configure and run 

## Versions
#### Cholesky
- cholesky_v1.3:   
A 1-D systolic array design for Cholesky Decomposition along projection vector (i,j,k)=(0,1,0) and (i,k)=(0,1), as illustrated below in (b). 
- cholesky_v4.0:   
A 1-D systolic array design for Cholesky Decomposition along projection vector (i,j,k)=(0,1,0) and (i,k)=(1,0), as illustrated below in (c). 
<img src="https://github.com/Sibylau/HLS_designs/blob/master/Illustrations/chol_1d.png" width="700">

- cholesky_v3.2:   
A 2-D systolic array design for Cholesky Decomposition along projection vector (i,j,k)=(1,0,0), as illustrated below in (b). 
<img src="https://github.com/Sibylau/HLS_designs/blob/master/Illustrations/chol_2d.png" width="580">

- cholesky_v2.2:   
A 1-D systolic array design for Cholesky Decomposition along projection vector (i,j,k)=(0,1,0) and (i,k)=(0,1), as illustrated below in (b). 
<img src="https://github.com/Sibylau/HLS_designs/blob/master/Illustrations/re_chol.png" width="360">

#### LU
- lu1D_v1.0:    
A 1-D systolic array design for LU Decomposition along projection vector (i,j,k)=(0,1,0) and (i,k)=(0,1), as illustrated at the bottom of the picture below. 

- lu1D_v2.0:    
A 1-D systolic array design for LU Decomposition along projection vector (i,j,k)=(0,1,0) and (i,k)=(1,0), as illustrated at the right of the picture below.
<img src="https://github.com/Sibylau/HLS_designs/blob/master/Illustrations/lu_1d.png" width="350">

- lu2D_v1.0:    
A 2-D systolic array design for LU Decomposition along projection vector (i,j,k)=(0,1,0), as illustrated below in (b).
<img src="https://github.com/Sibylau/HLS_designs/blob/master/Illustrations/lu_2d.png" width="500">

#### QR
- qr_v1.1:    
A 1-D systolic array design for QR Decomposition along projection vector (i,j,k)=(1,0,0) and (j,k)=(0,1), as illustrated at the bottom of the picture below. 

- qr_v1.2:    
Replace unroll with pipeline in qr_v1.1 for better performance, as automatically unrolled rotations will not be parallelized due to FIFO conficts. 
<img src="https://github.com/Sibylau/HLS_designs/blob/master/Illustrations/qr_1d.png" width="320">

- qr_v2.1:    
A 2-D systolic array design for QR Decomposition along projection vector (i,j,k)=(1,0,0), as illustrated below in (b). 
<img src="https://github.com/Sibylau/HLS_designs/blob/master/Illustrations/qr_2d.png" width="500">


