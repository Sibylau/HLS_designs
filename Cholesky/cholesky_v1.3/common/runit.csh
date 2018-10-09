#!/bin/csh

src_config ./T_gen.cpp -x cholesky.cfg.xml -o ./gen.csh
chmod +x ./gen.csh
./gen.csh
rm -f ./src_cfg_succeed
rm -f ./gen.csh


