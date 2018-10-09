#!/bin/csh
//; $SIZE = param_define("SIZE", 4);

mkdir -p ../design_files/`$SIZE`x`$SIZE`/
src_config ../template/T_lu.h -x lu.cfg.xml -o ../design_files/`$SIZE`x`$SIZE`/lu.h
rm -f ./src_cfg_succeed

src_config ../template/T_lu.cpp -x lu.cfg.xml -o ../design_files/`$SIZE`x`$SIZE`/lu.cpp
rm -f ./src_cfg_succeed

src_config ../template/T_lu_tb.cpp -x lu.cfg.xml -o ../design_files/`$SIZE`x`$SIZE`/lu_tb.cpp
rm -f ./src_cfg_succeed

cp -f ./script.tcl ../design_files/`$SIZE`x`$SIZE`/script.tcl
#cd ../design_files/`$SIZE`x`$SIZE`/
#chmod +x ../design_files/`$SIZE`x`$SIZE`/script.tcl
#vivado_hls ../design_files/`$SIZE`x`$SIZE`/script.tcl &

