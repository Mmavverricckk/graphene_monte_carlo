
## CODE FROM https://www.oreilly.com/library/view/bash-cookbook/0596526784/ch06s13.html
#!/bin/bash
#! /usr/bin/env python
echo Begun!
#for i in {1..5}; do

#python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 0.2 --J2 -0.001 --N 80

#python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 0.2 --J2 -0.001 --N 20

python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 0.0 --J2 0.0 --N 20

#python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 0.2 --J2 0.001 --N 20

#python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 0.2 --J2 -0.001 --N 20

#python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 0.0 --J2 0.0 --N 80

#python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 0.2 --J2 0.001 --N 400

#python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 0.2 --J2 0.001 --N 1000

#python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 0.2 --J2 -0.001 --N 200

#python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 0.2 --J2 -0.001 --N 500

#python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 0.2 --J2 0.001 --N 400
#python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 0.2 --J2 0.001 --N 1000

#python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 0.2 --J2 -0.001 --N 200
#python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 0.2 --J2 -0.001 --N 500

#python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 0.0 --J2 0.0 --N 80

#for i in $(seq 0.0 0.1 1); do
#for i in $(seq 0.5 0.1 1); do						# intraedge ferromagnetic strong coupling
#for i in $(seq 2 0.2 3); do						# intraedge superparamag case
#for i in $(seq 0.6 0.2 3); do						# intraedge superparamag case
#	for j in $(seq 0.0 0.001 0.005); do
#	for j in $(seq -0.0001 -0.0001 -0.0005); do		# interedge antiferromagnetic coupling 		(for ferrimag case)
#	for j in $(seq -0.001 -0.001 -0.005); do		# interedge antiferromagnetic coupling 		(for ferrimag case)
#	for j in $(seq -0.0005 0.0001 0.0005); do		# interedge ferromagnetic weak coupling 	(for ferromag case)
#	for j in $(seq 0.001 0.001 0.005); do			# interedge ferromagnetic weak coupling 	(for ferromag case)
#	for j in $(seq 0 1 0); do						# for superparamag case
#		python3 trans.py $i $j

#		python3 MH_add_and_plot.py --J1 $i --J2 $j --N 100
#		python3 MH_add_and_plot.py --J1 $i --J2 $j --N 50
#		python3 MH_add_and_plot.py --J1 $i --J2 $j --N 40

#		python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 $i --J2 $j --N 1000
#		python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 $i --J2 $j --N 400
#		python3 Combined_My_ising_1D_GNP_GNR.py --J1 0.2 --J2 0.001 --N 1000
#		python3 Combined_My_ising_1D_GNP_GNR.py --J1 0.2 --J2 0.001 --N 400

#		python3 Combined_My_ising_1D_GNP_GNR.py --J1 $i --J2 0.0 --N 100
#		python3 Combined_My_ising_1D_GNP_GNR.py --J1 $i --J2 0.0 --N 50
#		python3 Combined_My_ising_1D_GNP_GNR.py --J1 $i --J2 0.0 --N 40
#		python3 Combined_My_ising_1D_GNP_GNR.py --J1 $i --J2 0.0 --N 20


#		python3 Combined_My_ising_1D_GNP_GNR.py --J1 0.0 --J2 0.0 --N 8			# best SPM, with g = 16*2.002

#	done						
# done

#for i in $(seq 0.0 0.1 1); do
#	for j in $(seq -0.001 -0.001 -0.005); do
#		python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 $i --J2 $j --N 200
#		python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 $i --J2 $j --N 400
#		python3 /home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/Combined_My_ising_1D_GNP_GNR.py --J1 $i --J2 $j --N 500
#	done						
# done

