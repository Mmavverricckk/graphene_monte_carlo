#!/bin/bash
#! /usr/bin/env python
echo Bash script begun!
#for i in {1..5}; do
#for i in $(seq 0.1 0.1 0.5); do			
#for i in $(seq 2 0.2 3); do				
#for i in $(seq 0 0.5 3); do					
#	python3 MT_My_ising_1D_GNP_GNR.py --N 80 --J1 $i --J2 $j --J3 $k
#done
for i in $(seq 0.0 0.5 0.5); do					
#	python3 MT_My_ising_1D_GNP_GNR.py --N 80 --J1 0 --J2 0 --J3 $i
	python3 MT_My_ising_1D_GNP_GNR.py --N 400 --J1 0.2 --J2 0.001 --J3 $i
	python3 MT_My_ising_1D_GNP_GNR.py --N 1000 --J1 0.2 --J2 0.001 --J3 $i
#	python3 MT_My_ising_1D_GNP_GNR.py --N 200 --J1 0.2 --J2 -0.001 --J3 $i
#	python3 MT_My_ising_1D_GNP_GNR.py --N 500 --J1 0.2 --J2 -0.001 --J3 $i
done

echo Bash script ended!
