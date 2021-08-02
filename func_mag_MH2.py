"""
Monte Carlo simulation of the 1D Ising model
SOME OF THE ORIGINAL FUNCTIONS ARE KINDLY BORROWED FROM:
http://www.physics.rutgers.edu/~haule/681/src_MC/python_codes/ising.py
"""

import numpy as np
from scipy import *  # to define zeros((N), dtype=int) array
from parameters_MH import *

import os
######################### Function definitions: ###########################################################

def CEnergy(edge1, edge2, J1, J2, J3, T):  #J1 intraedge, J2 interedge, J3 external field
	"Energy of a 1D Ising lattice at particular configuration"
	Ene = 0	# Total energy = H1 + H2 + H3
	H1 = 0	# energy due to intraedge interactions
	H2 = 0	# energy due to interedge interactions
	H3 = 0	# energy due to interaction of individual spins with ext. mag. field
	N_ = len(edge1)
	for i in range(len(edge1)):
		H1 = -J1*k_B*T*edge1[i]*(edge1[(i-1)%N_] + edge1[(i+1)%N_])   # works becuase in general length of edge1 and edge2 is chosen SAME
	if N_ == 80:
		H3 = -16*J3*mu_B*g*edge1[i]
	else:
		H3 = -J3*mu_B*g*edge1[i]
	if J2 != 0:
		H2 = -J2*k_B*T*edge1[i]*sum(edge2)
	Ene = H1+H2+H3
	return Ene

def RandomL(N):
    "Radom lattice, corresponding to infinite temerature"
    edge1 = zeros((N), dtype=int)
    edge2 = zeros((N), dtype=int)
    for i in range(N):
            edge1[i] = np.sign(2*rand()-1)
            edge2[i] = np.sign(2*rand()-1)
    return (edge1, edge2)

def Equilibriate(warm, edge1, edge2, J1, J2, J3, T_high, T_divisor):
	T = T_high/T_divisor
	Ene = CEnergy(edge1, edge2, J1, J2, J3, T)	# Starting energy
	Mn = sum(edge1)	      		# Starting magnetization of edge
	N_ = len(edge1)
	for itt in range(warm):		# number MC steps to bring system into equilm have been achieved, 
		t = int(rand()*N_)   		# Randomly choosing a site to carry out energy calculation
		x = rand()
		if x < 0.5:
			(edge1, Ene, Mn) = flip_spin(edge1, edge2, J1, J2, J3, t, Ene, Mn, T)
		else:
			(edge2, Ene, Mn) = flip_spin(edge2, edge1, J1, J2, J3, t, Ene, Mn, T)
	return (edge1, edge2)

def flip_spin(edge1, edge2, J1, J2, J3, t, Ene, Mn, T):
	"flipping spin chosen (indexed t) from edge, after doing energy calc"
	edge2_sum = sum(edge2)	# summing the spins of the opposite edge
	N_ = len(edge1)
	S = edge1[t]
	if N_ == 80:
		del_E = 2*S*(k_B*T*((J1*(edge1[(t-1)%N_]+edge1[(t+1)%N_]) + J2*edge2_sum)) + 16*J3*g*mu_B)
	else:
		del_E = 2*S*(k_B*T*((J1*(edge1[(t-1)%N_]+edge1[(t+1)%N_]) + J2*edge2_sum)) + J3*g*mu_B)
	if del_E < 0:		# flip the spin if energy change is negative and update Energy and Magnetization
		edge1[t] = -S
		Ene += del_E
		Mn -= 2*S
	else:				# if energy change is positive flip with probability P = exp(-del_E/T)
		P = exp(-del_E/(k_B*T))
		if P>rand():	# flip the spin and update Energy and Magnetization
			edge1[t] = -S
			Ene += del_E
			Mn -= 2*S
	return (edge1, Ene, Mn)

def SamplePython(Nitt, edge1, edge2, J1, J2, J3, T):
	"Monte Carlo sampling for the Ising model in Pythons"
	Ene1 = CEnergy(edge1, edge2, J1, J2, J3, T)	# Starting energy
	Mn1 = sum(edge1)					# Starting magnetization for edge
	Ene2 = CEnergy(edge2, edge1, J1, J2, J3, T)	# Starting energy
	Mn2 = sum(edge2)					# Starting magnetization for edge
#    print([Ene, Mn])
	Naver=0.0    					# Measurements  (no. of measurements after warming MC steps)
	Eaver=0.0
	Maver=0.0
	Mn_sq_aver = 0.0
	Maver_abs = 0.0
	N_ = len(edge1)
	Mn_sq = 0
	for itt in range(Nitt):
		t = int(rand()*N_)   		# Randomly choosing a site to carry out energy calculation
		x = rand()
		if x < 0.5:
			(edge1, Ene1, Mn1) = flip_spin(edge1, edge2, J1, J2, J3, t, Ene1, Mn1, T)
		else:
			(edge2, Ene2, Mn2) = flip_spin(edge2, edge1, J1, J2, J3, t, Ene2, Mn2, T)
		if itt > warm and itt%measure == 0:   #itt%measure lets record data every measure no. of steps
			Naver += 1
			Eaver += Ene1/N_			# avg energy per spin for this iteration
			Maver += Mn1/N_			# avg magnetization per spin for this iteration
#			for kl in range(N_):	# sum of square of spin on each edge
#				Mn_sq += edge1[kl]*edge1[kl]
			Mn_sq += (Mn1*Mn1)/(N_*N_)
			Maver_abs += abs(Mn1)
	Chi = ((Mn_sq/(Naver))-(pow((Maver/Naver),2)))/(k_B*T)				# expectation value (per spin) of magnetization squared   	    	
	return (Maver/Naver, Eaver/Naver, Naver, Chi)

'''
#print([itt, Naver, Eaver, Maver])
#Maver_abs =  [abs(ele) for ele in Maver]
'''
'''
with open('data.csv', 'a') as fd:
writer = csv.writer(fd)
writer.writerow([itt, Ene, Mn])
'''

def MH_loop(Nitt, edge1, edge2, J1, J2, T, J3_divisor, J3_high, J3_low, J3_int, pc_status, pc_multiplier):
	wMag=[]
	wEne=[]
	wfield=[]   # J3 changes, i.e. external field changes
	wChi=[]
	for J3 in range(0, J3_high, J3_int):
		#edge1 = ones((N), dtype=int)
		#edge2 = ones((N), dtype=int)
		(aM, aE, aN, aChi) = SamplePython(Nitt, edge1, edge2, J1, J2, J3/J3_divisor, T)
		wMag.append(aM)
		wEne.append(aE)
		wfield.append(J3/J3_divisor)
		wChi.append(aChi)
		if J3 == J3_high-1:
			print(pc_status+(25*pc_multiplier))
			pc_status += 25*pc_multiplier
			#os.system('spd-say "twelve point five percent"')
			#os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
	for J3 in range(J3_high, 0, (-1*J3_int)):
		#edge1 = ones((N), dtype=int)
		#edge2 = ones((N), dtype=int)
		(aM, aE, aN, aChi) = SamplePython(Nitt, edge1, edge2, J1, J2, J3/J3_divisor, T)
		wMag.append(aM)
		wEne.append(aE)
		wfield.append(J3/J3_divisor)
		wChi.append(aChi)
		if J3 == 1:
			print(pc_status+(25*pc_multiplier))
			pc_status += 25*pc_multiplier
			#os.system('spd-say "seventy five percent"')
			#os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
	for J3 in range(0, J3_low, (-1*J3_int)):
		#edge1 = ones((N), dtype=int)
		#edge2 = ones((N), dtype=int)
		(aM, aE, aN, aChi) = SamplePython(Nitt, edge1, edge2, J1, J2, J3/J3_divisor, T)
		wMag.append(aM)
		wEne.append(aE)
		wfield.append(J3/J3_divisor)
		wChi.append(aChi)
		if J3 == J3_low+1:
			print(pc_status+(25*pc_multiplier))
			pc_status += 25*pc_multiplier
			#os.system('spd-say "eighty seven point five percent"')
			#os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
	for J3 in range(J3_low, J3_high, J3_int):				# this loop is till J3_high so that complete plot is formed
		#edge1 = ones((N), dtype=int)
		#edge2 = ones((N), dtype=int)
		(aM, aE, aN, aChi) = SamplePython(Nitt, edge1, edge2, J1, J2, J3/J3_divisor, T)
		wMag.append(aM)
		wEne.append(aE)
		wfield.append(J3/J3_divisor)
		wChi.append(aChi)
		if J3 == J3_high-1:
			print(pc_status+(25*pc_multiplier))
			pc_status += 25*pc_multiplier
			os.system('spd-say "check"')
			#os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
	return (wMag, wEne, wfield, wChi)

def MT_loop(Nitt, edge1, edge2, J1, J2, J3, T_high, T_low, T_int, T_divisor):
	wMag=[]
	wEne=[]
	wTemp=[]   # T changes, external field (J3) is held constant.
	wChi=[]
	# N1 N2 N3 N4 are used to show status of program on screen (percent finished):
	N1 = 0.75
	N2 = 0.50
	N3 = 0.25
	N4 = 1
	for T in range(T_high, T_low, (-1*T_int)):
		#edge1 = ones((N), dtype=int)
		#edge2 = ones((N), dtype=int)
		(aM, aE, aN, aChi) = SamplePython(Nitt, edge1, edge2, J1, J2, J3, T/T_divisor)
		wMag.append(aM)
		wEne.append(aE)
		wTemp.append(T/T_divisor)
		wChi.append(aChi)
		if T/(T_high-T_low) == 1-N1:
			print(str(N1*100)+'%')
		elif T/(T_high-T_low) == 1-N2:
			print(str(N2*100)+'%')
		elif T/(T_high-T_low) == 1-N3:
			print(str(N3*100)+'%')
		elif T/(T_high-T_low) == 1-N4:
			print(str(N4*100)+'%')
			os.system('spd-say "check"')
	return (wMag, wEne, wTemp, wChi)
