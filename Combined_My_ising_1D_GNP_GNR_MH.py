#!/usr/bin/env python
"""
Monte Carlo simulation of the 1D Ising model

http://www.physics.rutgers.edu/~haule/681/src_MC/python_codes/ising.py
"""
from matplotlib.pyplot import imshow
import numpy as np
from scipy import *
from pylab import *
import math as math
import sys # to write data to file
import csv # for handling csv files, data appending
import os
from os import path		# for checking if data file already exists or not
from numpy import array, savetxt # for saving csv file
from parameters_MH import *	# parameter definitions are stored in this file
from func_mag import *		# function definitions are stored in this file
import sys, getopt
'''
print(str(sys.argv))
print(sys.argv[0]) # prints python_script.py
print(sys.argv[2]) # prints var2
'''
#J1_2 = float(sys.argv[1])
#J2_2 = float(sys.argv[2])
filepath = '/home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/'
# USING ARGUMENT PARSER (argparse package) to use arguments (in case of more than 2 arguments passed from terminal to script):
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--J1', nargs = 1)
parser.add_argument('--J2', nargs = 1)
parser.add_argument('--N', nargs = 1)
#parser.add_argument('--J3', nargs = 1)
args = parser.parse_args()
#print(args)
J1_2 = float(args.J1[0])
J2_2 = float(args.J2[0])
#J3 = float(args.J3[0])
N_2 = int(args.N[0])
print('[J1, J2, N] = ', [J1_2, J2_2, N_2])
#print('[J1, J2, J3] = ', [J1_2, J2_2, J3])
if N_2 == 80:
	g = 16*2.002
#print("J1_2 = ", str(J1_2), "J2_2 = ", str(J2_2), "Nitt = "+str(Nitt), "N = "+str(N_2))
#print("Started Calculation")

#########################################################################################################################
if g > 2.002:
	flag_spm = True      	# if 'true' then calculation is to be done for Superparamagnetic case
else:
	flag_spm = False

if J2_2 >= 0:
	FERRO = True
elif J2_2 < 0:
	FERRO = False
FERRI = 1-FERRO
#########################################################################################################################
if flag_spm == True:
	system_name = 'superparamag'
	Mag_label = r'M$_{superparamag}$'
	plot_color = ['black', 'red']
	file_flag = 0
elif flag_spm == False and FERRO == True:
	system_name = 'ferromag'
	Mag_label = r'M$_{ferromag}$'
	plot_color = ['blue', 'green']
	file_flag = 1
elif flag_spm == False and FERRI == True:
	system_name = 'ferrimag'
	Mag_label = r'M$_{ferrimag}$'
	plot_color = ['brown', 'navy']
	file_flag = 2
#############################################################################################################
print(filepath+"results/data_"+system_name+"_T"+str(T)+"_Nitt"+str(Nitt)+"_warmsteps"+str(warm)+"_measure"+str(measure)+"_N_2"+str(N_2)+"_J1_2"+str(J1_2)+"_J2_2"+str(J2_2)+".csv")
if path.exists(filepath+"results/data_"+system_name+"_T"+str(T)+"_Nitt"+str(Nitt)+"_warmsteps"+str(warm)+"_measure"+str(measure)+"_N_2"+str(N_2)+"_J1_2"+str(J1_2)+"_J2_2"+str(J2_2)+".csv") == False:

	(edge1, edge2) = RandomL(N_2)
	#edge1 = ones((N), dtype=int)
	#edge2 = ones((N), dtype=int)
	pc_status = 0
	pc_multiplier = 1 #multiplies by 2 the total values in case of Ferrimagnetic case
	(wMag, wEne, wfield, wChi) = MH_loop(Nitt, edge1, edge2, J1_2, J2_2, T, J3_divisor, J3_high, J3_low, J3_int, pc_status, pc_multiplier)
	f = open(filepath+"results/data_"+system_name+"_T"+str(T)+"_Nitt"+str(Nitt)+"_warmsteps"+str(warm)+"_measure"+str(measure)+"_N_2"+str(N_2)+"_J1_2"+str(J1_2)+"_J2_2"+str(J2_2)+".csv", "w")
	f.write("{},{},{},{}\n".format("Field", "Energy", "Magnetization", "Susceptibility"))
	for x in zip(wfield, wEne, wMag, wChi):
		f.write("{},{},{},{}\n".format(x[0], x[1], x[2], x[3]))
	f.close()
		#print([aE, aM])
######################################### PLOTTING COMMANDS #############################################
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import csv
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import codecs
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy import optimize
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import matplotlib.ticker as ticker

filepath_MH=['/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_detritus/PPMS_BVDGC/BVDGC5K.dat', '/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_detritus/PPMS_a-BVDGC/aBVDGC5K.dat']
filepath_deltaMH=['/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_detritus/PPMS_BVDGC/MH_BVDGC_Diam_subt.csv', '/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_detritus/PPMS_a-BVDGC/MH_aBVDGC_Diam_subt.csv']
file_name = ['superparamag', 'ferromag', 'ferrimag']

####################################################################################################
#########################  	 DATA ACQUISITION AND PLOTTING: 	####################################
####################################################################################################
mu_0 = 4*math.pi*(10**(-7))
############################## 	DATA ACQUISITION from files: 	####################################
num_skipped = 1  # no. of rows skipped from top
#Ha, Ma = np.loadtxt('./results/data_'+file_name[0]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_superparamag)+'_J1_2'+str(J1_superparamag)+'_J2_2'+str(J2_superparamag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
#Hb, Mb = np.loadtxt('./results/data_'+file_name[1]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferromag)+'_J1_2'+str(J1_ferromag)+'_J2_2'+str(J2_ferromag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
Hc, Mc = np.loadtxt(filepath+'results/data_'+file_name[file_flag]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_2)+'_J1_2'+str(J1_2)+'_J2_2'+str(J2_2)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

'''
Ha = (Ba/mu_0) - Ma   # units of H are Am-1   1 Am-1 = 0.01256637061436 Oe
Hb = (Bb/mu_0) - Mb   # units of H are Am-1   1 Am-1 = 0.01256637061436 Oe
Hc = (Bc/mu_0) - Mc   # units of H are Am-1   1 Am-1 = 0.01256637061436 Oe
'''
################################		 M-H data		###############################
#x, y = np.loadtxt(filepath_MH, dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2])

################################  	(DELTA)M-H data	    ###############################
B, delM = np.loadtxt(filepath_deltaMH[1], dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 1])
## convert B to H:
H = (B/mu_0) - delM   # units of H are Am-1   1 Am-1 = 0.01256637061436 Oe
#Hc = (Bc/mu_0) - Mc

#################### PLOTTING FROM DATA ACQUIRED from files: ##########################
divisor = ((max(Mc)-min(Mc))*(mu_B*N_Avo/N_2))/(max(delM)-min(delM))

fig, ax = plt.subplots()
plt.plot(H, delM, '-', color='red', label='aBVDGC exp at 5K')
plt.plot(10e5*Hc, (Mc*mu_B*N_Avo/N_2)/divisor, '-', color='navy', label='ferri')
#plt.plot(10e5*Ha, ((fa*Ma*mu_B*N_Avo/(N_superparamag*N_superparamag))+(fb*Mb*mu_B*N_Avo/(N_ferromag*N_ferromag))+(fc*Mc*mu_B*N_Avo/(N_ferrimag*N_ferrimag)))/divisor, '-', color='navy', label='sum of signals')
plt.xlim()
plt.ylim()
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#plt.legend()
plt.xlabel(r"$H$/A m$^{-1}$")
plt.ylabel(r"$\Delta$""$M$/emu g$^{-1}}$.")
plt.text(0, -0.02, [J1_2, J2_2, N_2, divisor], fontsize=12)
plt.legend()
#plt.show()
fig.savefig(filepath+"results/Exp_and_Fit_Plot"+"_Nitt"+str(Nitt)+"_"+str(warm)+"_"+str(measure)+"_"+str(N_2)+"_"+str(J1_2)+"_"+str(J2_2)+".png", format="png",dpi=600)
plt.close()

'''
x, y = np.loadtxt('./results/data_'+system_name+'_T'+str(T)+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_2)+'_J1_2'+str(J1_2)+'_J2_2'+str(J2_2)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
fig, ax = plt.subplots()

plt.plot(x, y*mu_B*N_Avo/(N_2*N_2), '-', color='blue', label=system_name)  # Change the label name as per system: superparamag, ferro, ferri
plt.legend()
fig.savefig("./results/plot_"+system_name+"_T"+str(T)+"_Nitt"+str(Nitt)+"_warmsteps"+str(warm)+"_measure"+str(measure)+"_N_2"+str(N_2)+"_J1_2"+str(J1_2)+"_J2_2"+str(J2_2)+".png", format="png",dpi=600)

plt.close()

'''











############################################## EXTRA CODE: ##########################################
'''
#################  TEMPERATURE and ARRAY SIZE input commands: ##############################

N = int(input('N_superparamag = '))	#10      # linear dimension of the 1st (Superparamag) lattice, lattice-size= N x N
N_2 = int(input('N_ferromag = ')) #2000        # linear dimension of the 2nd (Ferromag) lattice components, lattice-size= N_2 x N_2
T = int(input('T = '))
'''

'''
file = open("./results/data_superparamag_Nitt"+str(Nitt)+"_T"+str(T)+"_N"+str(N)+"_J1"+str(J1)+"_J2"+str(J2)+".csv", "r")
line_count = 0
for line in file:
    if line != "\n":
        line_count += 1
file.close()
print(line_count)
'''
'''
_N_rows = 500   # No. of max rows from top to be selected for plotting
num_skipped = 1  # no. of rows skipped from top
'''

#x, y = np.loadtxt('data.csv', dtype='double', delimiter=',', skiprows=0, max_rows=line_count, unpack=True, usecols=[0, 1])

#x, y = np.loadtxt('data.csv', dtype='double', delimiter=',', skiprows=num_skipped, max_rows=line_count-num_skipped, unpack=True, usecols=[0, 1])
#x, y1_5K = np.loadtxt('data_superparamag_'+str(T)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, max_rows=line_count-num_skipped, unpack=True, usecols=[0, 2])

'''
f1 = 0.05			# fraction of signal from each measurement to be added to get final signal
f2 = 1-f1
'''



'''
fig, ax = plt.subplots()
if N !=0 and N_2 !=0:
	plt.plot(x, ((f1*y1_5K*mu_B*N_Avo/(N*N)) + (f2*y2_5K*mu_B*N_Avo/(N_2*N_2)))/divisor, '-', color='black', label='combined SPM-Ferro')
	plt.xlim(xlim_low, xlim_high)
	plt.ylim(ylim_low, ylim_high)
	plt.legend()
	#plt.show()
	#plt.plot(x, y, 'r')		# Energy
	#plt.plot(x, y2, 'b')		#Magnetization
	fig.savefig("./results/Plot_SPM-Ferro"+"_T"+str(T)+"_Nitt"+str(Nitt)+"_warmsteps"+str(warm)+"_measure"+str(measure)+"_T"+str(T)+"_N"+str(N)+"_N_2"+str(N_2)+"_J1"+str(J1)+"_J2"+str(J2)+"_J1_2"+str(J1_2)+"_J2_2"+str(J2_2)+".png", format="png",dpi=600)

if N != 0:
	plt.plot(x, y1_5K*mu_B*N_Avo/(N*N), '-', color='red', label='superparamag')
	plt.legend()
	fig.savefig("./results/Plot_SPM"+"_T"+str(T)+"_Nitt"+str(Nitt)+"_warmsteps"+str(warm)+"_measure"+str(measure)+"_T"+str(T)+"_N"+str(N)+"_N_2"+str(N_2)+"_J1"+str(J1)+"_J2"+str(J2)+"_J1_2"+str(J1_2)+"_J2_2"+str(J2_2)+".png", format="png",dpi=600)
	#plt.show()
#plt.plot(x, y, '-', color='blue', label='Energy')

if N == 0 and FERRO == True:
	plt.plot(x, y2_5K*mu_B*N_Avo/(N_2*N_2), '-', color='blue', label='ferromag')
	plt.legend()
	fig.savefig("./results/Plot_Ferro"+"_T"+str(T)+"_Nitt"+str(Nitt)+"_warmsteps"+str(warm)+"_measure"+str(measure)+"_T"+str(T)+"_N"+str(N)+"_N_2"+str(N_2)+"_J1"+str(J1)+"_J2"+str(J2)+"_J1_2"+str(J1_2)+"_J2_2"+str(J2_2)+".png", format="png",dpi=600)
	#plt.show()

if N == 0 and FERRI == True:
	plt.plot(x, y2_5K*mu_B*N_Avo/(N_2*N_2), '-', color='green', label='ferrimag')
	plt.legend()
	fig.savefig("./results/Plot_Ferri"+"_T"+str(T)+"_Nitt"+str(Nitt)+"_warmsteps"+str(warm)+"_measure"+str(measure)+"_N"+str(N)+"_N_2"+str(N_2)+"_J1"+str(J1)+"_J2"+str(J2)+"_J1_2"+str(J1_2)+"_J2_2"+str(J2_2)+".png", format="png",dpi=600)
	#plt.show()
'''

#plt.text(2000, 5000, r'$\frac{I_{D_{Am}}}{I_{G_{Am}}} = $', fontsize=20)
