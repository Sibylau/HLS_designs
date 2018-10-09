
####
open_project chol
set_top cholesky_top
add_files cholesky.cpp
add_files cholesky.h
add_files -tb cholesky_tb.cpp
open_solution "solution1"
set_part {xc7vx690tffg1157-2} -tool vivado
create_clock -period 4 -name default

csim_design -compiler gcc
csynth_design
cosim_design -compiler gcc -trace_level all
#export_design -format ip_catalog

