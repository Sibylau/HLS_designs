#!/bin/csh
//; $SIZE = param_define("SIZE", 4);

mkdir -p ../design_files/`$SIZE`x`$SIZE`/
src_config ../template/T_chol.h -x cholesky.cfg.xml -o ../design_files/`$SIZE`x`$SIZE`/chol.h
rm -f ./src_cfg_succeed

src_config ../template/T_chol.cpp -x cholesky.cfg.xml -o ../design_files/`$SIZE`x`$SIZE`/chol.cpp
rm -f ./src_cfg_succeed

src_config ../template/T_chol_tb.cpp -x cholesky.cfg.xml -o ../design_files/`$SIZE`x`$SIZE`/chol_tb.cpp
rm -f ./src_cfg_succeed

cp -f ./script.tcl ../design_files/`$SIZE`x`$SIZE`/script.tcl
#cd ../design_files/`$SIZE`x`$SIZE`/
#chmod +x ../design_files/`$SIZE`x`$SIZE`/script.tcl
#vivado_hls ../design_files/`$SIZE`x`$SIZE`/script.tcl &

