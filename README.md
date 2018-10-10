# HLS_designs

Systolic array implementations for Cholesky, LU, and QR decomposition using HLS

## Get started
#### Environment
- Ubuntu 16.04.5 LTS
- Xilinx Vivado HLS v2017.4
- Matlab R2017a

#### Directory Tree

Inside each design folder, here are:
```
|-- Design_Folder/
  |- common/ 
  |- model4x4/
  |- template/
```
Folder `common/` includes script files shared for different designs.     
Folder `model4x4/` gives an example of 4x4 implementation, with detailed comments alongside the codes.     
Folder `template/` includes template cpp files used for generating codes.     
For a understanding of each design, please go to `model4x4/` and view the comments in `design_name.cpp`, and refer to the illustrations shown below if necessary : )


#### Run it
For each design,
- Go to `common/`. Find `algorithm_name.cfg.xml`, revise it according to your matrix size MxN.     
Please manually modify the parameter `BIT` according to `BIT = ceiling(log2(SIZE))`.
- Run `runit.csh`. It will generate a new folder `design_files/` with the design `MxN/` inside:
```
|-- Design_Folder/
  |- common/
  |- design_files/
    |- MxN/
  |- model4x4/
  |- template/
```
- Go to `Design_Folder/` and call `genA()` in `MATLAB` to generate a random operand matrix A required for testbench.      
Some specific cases like 8x8, 16x16 etc. are provided under `Design_Folder/`. You can use it or generate a new one.
```
How to use function genA():
For Cholesky: generating NxN symmetric positive definite matrix by calling genA(N)
For LU: generating NxN full rank matrix by calling genA(N)
For QR: generating MxN full rank matrix by calling genA(M,N)
```
- Go to `Design_Folder/design_files/MxN/`, run `script.tcl` in vivado_hls environment by `$vivado_hls script.tcl`.
- Revise `script.tcl` according to your demands. It by default runs through csim, synthesis and cosim.


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


