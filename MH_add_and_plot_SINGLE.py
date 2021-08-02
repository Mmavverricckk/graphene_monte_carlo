import numpy as np
import scipy as scipy
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
import math
from parameters_MH import *



# BVDGC is SPM+Ferro+Ferri
# aBVDGC is pure Ferro + Ferri
'''
######## best paramaters for aBVDGC (ferro+ferri):
J1_superparamag = 1.7
J2_superparamag = 0.0
J1_ferromag = 0.3
J2_ferromag = 0.0001
J1_ferrimag = 0.4
J2_ferrimag = -0.002
N_superparamag = 50
N_ferromag = 200
N_ferrimag = 200
fa = 0.0
fb = 0.001
fc = 1-fa-fb		
divisor = 1
filepath_MH='/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_detritus/PPMS_a-BVDGC/aBVDGC5K.dat'
filepath_deltaMH='/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_detritus/PPMS_a-BVDGC/MH_aBVDGC_Diam_subt.csv'
'''

'''
######## best paramaters for BVDGC (spm+ferro+ferri):
J1_superparamag = 1.7
J2_superparamag = 0.0
J1_ferromag = 0.3
J2_ferromag = 0.003
J1_ferrimag = 0.4
J2_ferrimag = -0.001
N_superparamag = 50
N_ferromag = 200
N_ferrimag = 200
fa = 0.02
fb = 0.04
fc = 1-fa-fb		
divisor = 5
filepath_MH='/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_detritus/PPMS_BVDGC/BVDGC5K.dat'
filepath_deltaMH='/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_detritus/PPMS_BVDGC/MH_BVDGC_Diam_subt.csv'
'''

filepath_MH='/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_detritus/PPMS_BVDGC/BVDGC5K.dat'
filepath_deltaMH='/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_detritus/PPMS_a-BVDGC/MH_aBVDGC_Diam_subt.csv'


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--J1', nargs = 1)
parser.add_argument('--J2', nargs = 1)
#parser.add_argument('--J3', nargs = 1)
args = parser.parse_args()
#print(args)
J1_2 = float(args.J1[0])
J2_2 = float(args.J2[0])
#J3 = float(args.J3[0])
print('[J1, J2] = ', [J1_2, J2_2])
#print('[J1, J2, J3] = ', [J1_2, J2_2, J3])

#########################################################################################################################
if N_2 <= 50:
	flag_spm = True      	# if 'true' then calculation is to be done for Superparamagnetic case
else:
	flag_spm = False

if J2_2 > 0:
	FERRO = True
elif J2_2 <= 0:
	FERRO = False
FERRI = 1-FERRO
#########################################################################################################################
if flag_spm == True:
	system_name = 'superparamag'
	Mag_label = r'M$_{superparamag}$'
	plot_color = ['black', 'red']
elif flag_spm == False and FERRO == True:
	system_name = 'ferromag'
	Mag_label = r'M$_{ferromag}$'
	plot_color = ['blue', 'green']
elif flag_spm == False and FERRI == True:
	system_name = 'ferrimag'
	Mag_label = r'M$_{ferrimag}$'
	plot_color = ['brown', 'navy']

######################################################################################################################
#########################  	 DATA ACQUISITION AND PLOTTING: 	######################################################
######################################################################################################################
mu_0 = 4*math.pi*(10**(-7))
################## 	DATA ACQUISITION from files: 	#########################
num_skipped = 1  # no. of rows skipped from top
Ha, Ma = np.loadtxt('./results/data_'+system_name+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_2)+'_J1_2'+str(J1_2)+'_J2_2'+str(J2_2)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
print('./results/data_'+system_name+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_2)+'_J1_2'+str(J1_2)+'_J2_2'+str(J2_2)+'.csv')

#Ha = (Ba/mu_0) - Ma   # units of H are Am-1   1 Am-1 = 0.01256637061436 Oe

##########################		 M-H data				#########################
#x, y = np.loadtxt(filepath_MH, dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2])

##########################  	(DELTA)M-H data			   ######################
B, M = np.loadtxt(filepath_MH, dtype='double', delimiter='\t', skiprows=1, unpack=True, usecols=[0, 2])
#B2, delM = np.loadtxt(filepath_deltaMH, dtype='double', delimiter=',', skiprows=1, max_rows=len(M), unpack=True, usecols=[0, 1])
B2, delM = np.loadtxt(filepath_deltaMH, dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 1])
## convert B to H:
#H = (B/mu_0) - M   # units of H are Am-1   1 Am-1 = 0.01256637061436 Oe
H = (B2/mu_0) - delM   # units of H are Am-1   1 Am-1 = 0.01256637061436 Oe
################# PLOTTING FROM DATA ACQUIRED from files: #######################
fig, ax = plt.subplots()
plt.plot(H, delM, 'o', color='cyan', label='aBVDGC exp at 5K')
plt.plot(Ha*10e5, Ma*mu_B*N_Avo/(N_2*N_2*1.5), '-', color='navy', label=system_name+'signal')
plt.xlim()
plt.ylim()
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#plt.legend()
plt.xlabel(r"$H$/A m$^{-1}$")
plt.ylabel(r"$\Delta$""$M$/emu g$^{-1}}$.")
plt.show()
plt.close()

