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

'''
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
'''


J1_superparamag = 1.7
J2_superparamag = 0.0
J1_ferromag = 0.3
J2_ferromag = 0.0001
J1_ferrimag = 0.2
J2_ferrimag = -0.004
N_superparamag = 50
N_ferromag = 200
N_ferrimag = 100
fa = 0.0
fb = 0.001
fc = 1-fa-fb		
divisor = 1
filepath_MH='/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_detritus/PPMS_a-BVDGC/aBVDGC5K.dat'
filepath_deltaMH='/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_detritus/PPMS_a-BVDGC/MH_aBVDGC_Diam_subt.csv'
file_name = ['superparamag', 'ferromag', 'ferrimag']
######################################################################################################################
#########################  	 DATA ACQUISITION AND PLOTTING: 	######################################################
######################################################################################################################
mu_0 = 4*math.pi*(10**(-7))
################## 	DATA ACQUISITION from files: 	#########################
num_skipped = 1  # no. of rows skipped from top
Ba, Ma = np.loadtxt('./results/data_'+file_name[0]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_superparamag)+'_J1_2'+str(J1_superparamag)+'_J2_2'+str(J2_superparamag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
Bb, Mb = np.loadtxt('./results/data_'+file_name[1]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferromag)+'_J1_2'+str(J1_ferromag)+'_J2_2'+str(J2_ferromag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
Bc, Mc = np.loadtxt('./results/data_'+file_name[2]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferrimag)+'_J1_2'+str(J1_ferrimag)+'_J2_2'+str(J2_ferrimag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

Ha = (Ba/mu_0) - Ma   # units of H are Am-1   1 Am-1 = 0.01256637061436 Oe
Hb = (Bb/mu_0) - Mb   # units of H are Am-1   1 Am-1 = 0.01256637061436 Oe
Hc = (Bc/mu_0) - Mc   # units of H are Am-1   1 Am-1 = 0.01256637061436 Oe

##########################		 M-H data				#########################
#x, y = np.loadtxt(filepath_MH, dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2])

##########################  	(DELTA)M-H data			   ######################
B, delM = np.loadtxt(filepath_deltaMH, dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 1])
## convert B to H:
H = (B/mu_0) - delM   # units of H are Am-1   1 Am-1 = 0.01256637061436 Oe
################# PLOTTING FROM DATA ACQUIRED from files: #######################
fig, ax = plt.subplots()

plt.plot(H, delM, 'o', color='cyan', label='aBVDGC exp at 5K')
plt.plot(Ha, ((fa*Ma*mu_B*N_Avo/(N_superparamag*N_superparamag))+(fb*Mb*mu_B*N_Avo/(N_ferromag*N_ferromag))+(fc*Mc*mu_B*N_Avo/(N_ferrimag*N_ferrimag)))/divisor, '-', color='navy', label='sum of signals')
plt.xlim()
plt.ylim()
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#plt.legend()
plt.xlabel(r"$H$/A m$^{-1}$")
plt.ylabel(r"$\Delta$""$M$/emu g$^{-1}}$.")
fig.savefig("./results/aBVDGC_summed_plot"+"_Nitt"+str(Nitt)+"_"+str(warm)+"_"+str(measure)+"_spmfr_"+str(fa)+"fmfr_"+str(fb)+"fimfr_"+str(fc)+"_"+str(J1_superparamag)+"_"+str(J2_superparamag)+"_"+str(J1_ferromag)+"_"+str(J2_ferromag)+"_"+str(J1_ferrimag)+"_"+str(J2_ferrimag)+"_"+str(N_superparamag)+"_"+str(N_ferromag)+"_"+str(N_ferrimag)+".png", format="png",dpi=600)
plt.show()
plt.close()

