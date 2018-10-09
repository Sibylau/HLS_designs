#!/bin/csh
//; $ROW = param_define("ROW", 4);
//; $COL = param_define("COL", 3);

mkdir -p ../design_files/`$ROW`x`$COL`/
src_config ../template/T_qr.h -x qr.cfg.xml -o ../design_files/`$ROW`x`$COL`/qr.h
rm -f ./src_cfg_succeed

src_config ../template/T_qr.cpp -x qr.cfg.xml -o ../design_files/`$ROW`x`$COL`/qr.cpp
rm -f ./src_cfg_succeed

src_config ../template/T_qr_tb.cpp -x qr.cfg.xml -o ../design_files/`$ROW`x`$COL`/qr_tb.cpp
rm -f ./src_cfg_succeed

cp -f ./script.tcl ../design_files/`$ROW`x`$COL`/script.tcl
#cd ../design_files/`$ROW`x`$COL`/
#chmod +x ../design_files/`$ROW`x`$COL`/script.tcl
#vivado_hls ../design_files/`$ROW`x`$COL`/script.tcl &

