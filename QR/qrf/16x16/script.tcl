
####
open_project qr
set_top qrf_top
add_files qrf.cpp
add_files qrf.h
add_files -tb qrf_tb.cpp
open_solution "solution1"
set_part {xc7vx690tffg1157-2} -tool vivado
create_clock -period 4 -name default

csim_design -compiler gcc
csynth_design
cosim_design -compiler gcc -trace_level all
#export_design -format ip_catalog

