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

###########################################################################################

filepath = '/home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/'

#############################################################################################################

filepath_MH=['/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_detritus/PPMS_BVDGC/BVDGC5K.dat', '/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_detritus/PPMS_a-BVDGC/aBVDGC5K.dat']

filepath_deltaMH=['/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_petal/PPMS_BVPGC/MH_BVPGC_Diam_subt.csv', '/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_petal/PPMS_aBVPGC/MH_aBVPGC_Diam_subt.csv', '/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_detritus/PPMS_BVDGC/MH_BVDGC_Diam_subt.csv', '/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_detritus/PPMS_a-BVDGC/MH_aBVDGC_Diam_subt.csv']

file_name = ['superparamag', 'ferromag', 'ferrimag']
###########################################################################################

mu_0 = 4*math.pi*(10**(-7))


fig, ax = plt.subplots(2,2)












###########################################################################################

######## best paramaters for BVPGC (spm+ferro+ferri):

J1_superparamag = 0.0
J2_superparamag = 0.0
N_superparamag = 80

J1_ferromag = 0.2
J2_ferromag = 0.001
N_ferromag1 = 400
N_ferromag2 = 1000

J1_ferrimag = J1_ferromag
J2_ferrimag = -0.001
N_ferrimag1 = 200
N_ferrimag2 = 500

fspm = 0.1
fferro1 = 0.4
fferro2 = 0.01 #0.03
fferri1 = 0.2
fferri2 = 1-(fspm + fferro1 + fferro2 + fferri1)


############################## 	DATA ACQUISITION from files: 	####################################
num_skipped = 1  # no. of rows skipped from top
H_spm, M_spm = np.loadtxt(filepath+'./results/data_'+file_name[0]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_superparamag)+'_J1_2'+str(J1_superparamag)+'_J2_2'+str(J2_superparamag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

H_ferro1, M_ferro1 = np.loadtxt(filepath+'./results/data_'+file_name[1]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferromag1)+'_J1_2'+str(J1_ferromag)+'_J2_2'+str(J2_ferromag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
H_ferro2, M_ferro2 = np.loadtxt('./results/data_'+file_name[1]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferromag2)+'_J1_2'+str(J1_ferromag)+'_J2_2'+str(J2_ferromag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

H_ferri1, M_ferri1 = np.loadtxt(filepath+'results/data_'+file_name[2]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferrimag1)+'_J1_2'+str(J1_ferrimag)+'_J2_2'+str(J2_ferrimag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
H_ferri2, M_ferri2 = np.loadtxt(filepath+'results/data_'+file_name[2]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferrimag2)+'_J1_2'+str(J1_ferrimag)+'_J2_2'+str(J2_ferrimag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

################################ 	   M-H data         ###############################
#x, y = np.loadtxt(filepath_MH, dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2])

################################  	(DELTA)M-H data	    ###############################
B, delM = np.loadtxt(filepath_deltaMH[0], dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 1])
## convert B to H:
H = (B/mu_0) - delM   # units of H are Am-1   1 Am-1 = 0.01256637061436 Oe
#Hc = (Bc/mu_0) - Mc

#################### PLOTTING FROM DATA ACQUIRED from files: ##########################

divisor_spm = ((max(M_spm)-min(M_spm))*(mu_B*N_Avo/N_superparamag))/(max(delM)-min(delM))
divisor_ferro1 = ((max(M_ferro1)-min(M_ferro1))*(mu_B*N_Avo/N_ferromag1))/(max(delM)-min(delM))
divisor_ferro2 = ((max(M_ferro2)-min(M_ferro2))*(mu_B*N_Avo/N_ferromag2))/(max(delM)-min(delM))
divisor_ferri1 = ((max(M_ferri1)-min(M_ferri1))*(mu_B*N_Avo/N_ferrimag1))/(max(delM)-min(delM))
divisor_ferri2 = ((max(M_ferri2)-min(M_ferri2))*(mu_B*N_Avo/N_ferrimag2))/(max(delM)-min(delM))
'''
divisor_spm = 1
divisor_ferro1 = 1
divisor_ferro2 = 1
divisor_ferri1 = 1
divisor_ferri2 = 1
'''

ax[0,0].plot(H, delM, '-', color='black', label='BVPGC exp at 5K')
ax[0,0].plot(10e5*H_ferri2, (fspm*M_spm*mu_B*N_Avo/N_superparamag)/divisor_spm+(fferro1*M_ferro1*mu_B*N_Avo/N_ferromag1)/divisor_ferro1+(fferro2*M_ferro2*mu_B*N_Avo/N_ferromag2)/divisor_ferro2+(fferri1*M_ferri1*mu_B*N_Avo/N_ferrimag1)/divisor_ferri1+(fferri2*M_ferri2*mu_B*N_Avo/N_ferrimag2)/divisor_ferri2, '-', color='red', label='fit')
#plt.plot(10e5*Ha, ((fa*Ma*mu_B*N_Avo/(N_superparamag*N_superparamag))+(fb*Mb*mu_B*N_Avo/(N_ferromag*N_ferromag))+(fc*Mc*mu_B*N_Avo/(N_ferrimag*N_ferrimag)))/divisor, '-', color='navy', label='sum of signals')
########################################################################################################################









###########################################################################################

######## best paramaters for a-BVPGC (spm+ferro+ferri):

J1_superparamag = 0.0
J2_superparamag = 0.0
N_superparamag = 20

J1_ferromag = 0.2
J2_ferromag = 0.001
N_ferromag1 = 400
N_ferromag2 = 1000

J1_ferrimag = J1_ferromag
J2_ferrimag = -0.001
N_ferrimag1 = 200
N_ferrimag2 = 500

fspm = 0.9
fferro1 = 0.1
fferro2 = 0.0
fferri1 = 0.0
fferri2 = 0.0 #1-(fspm + fferro1 + fferro2 + fferri1)


############################## 	DATA ACQUISITION from files: 	####################################
num_skipped = 1  # no. of rows skipped from top
H_spm, M_spm = np.loadtxt(filepath+'./results/data_'+file_name[0]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_superparamag)+'_J1_2'+str(J1_superparamag)+'_J2_2'+str(J2_superparamag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

H_ferro1, M_ferro1 = np.loadtxt(filepath+'./results/data_'+file_name[1]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferromag1)+'_J1_2'+str(J1_ferromag)+'_J2_2'+str(J2_ferromag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
H_ferro2, M_ferro2 = np.loadtxt('./results/data_'+file_name[1]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferromag2)+'_J1_2'+str(J1_ferromag)+'_J2_2'+str(J2_ferromag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

H_ferri1, M_ferri1 = np.loadtxt(filepath+'results/data_'+file_name[2]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferrimag1)+'_J1_2'+str(J1_ferrimag)+'_J2_2'+str(J2_ferrimag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
H_ferri2, M_ferri2 = np.loadtxt(filepath+'results/data_'+file_name[2]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferrimag2)+'_J1_2'+str(J1_ferrimag)+'_J2_2'+str(J2_ferrimag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

################################ 	   M-H data         ###############################
#x, y = np.loadtxt(filepath_MH, dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2])

################################  	(DELTA)M-H data	    ###############################
B, delM = np.loadtxt(filepath_deltaMH[1], dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 1])
## convert B to H:
H = (B/mu_0) - delM   # units of H are Am-1   1 Am-1 = 0.01256637061436 Oe
#Hc = (Bc/mu_0) - Mc

#################### PLOTTING FROM DATA ACQUIRED from files: ##########################

divisor_spm = ((max(M_spm)-min(M_spm))*(mu_B*N_Avo/N_superparamag))/(max(delM)-min(delM))
divisor_ferro1 = ((max(M_ferro1)-min(M_ferro1))*(mu_B*N_Avo/N_ferromag1))/(max(delM)-min(delM))
divisor_ferro2 = ((max(M_ferro2)-min(M_ferro2))*(mu_B*N_Avo/N_ferromag2))/(max(delM)-min(delM))
divisor_ferri1 = ((max(M_ferri1)-min(M_ferri1))*(mu_B*N_Avo/N_ferrimag1))/(max(delM)-min(delM))
divisor_ferri2 = ((max(M_ferri2)-min(M_ferri2))*(mu_B*N_Avo/N_ferrimag2))/(max(delM)-min(delM))
'''
divisor_spm = 1
divisor_ferro1 = 1
divisor_ferro2 = 1
divisor_ferri1 = 1
divisor_ferri2 = 1
'''

ax[0,1].plot(H, delM, '-', color='black', label='a-BVPGC exp at 5K')
ax[0,1].plot(10e5*H_ferri2, (fspm*M_spm*mu_B*N_Avo/N_superparamag)/divisor_spm+(fferro1*M_ferro1*mu_B*N_Avo/N_ferromag1)/divisor_ferro1+(fferro2*M_ferro2*mu_B*N_Avo/N_ferromag2)/divisor_ferro2+(fferri1*M_ferri1*mu_B*N_Avo/N_ferrimag1)/divisor_ferri1+(fferri2*M_ferri2*mu_B*N_Avo/N_ferrimag2)/divisor_ferri2, '-', color='red', label='fit')
#plt.plot(10e5*Ha, ((fa*Ma*mu_B*N_Avo/(N_superparamag*N_superparamag))+(fb*Mb*mu_B*N_Avo/(N_ferromag*N_ferromag))+(fc*Mc*mu_B*N_Avo/(N_ferrimag*N_ferrimag)))/divisor, '-', color='navy', label='sum of signals')
########################################################################################################################












###########################################################################################

######## best paramaters for BVDGC (spm+ferro+ferri):

J1_superparamag = 0.0
J2_superparamag = 0.0
N_superparamag = 80

J1_ferromag = 0.2
J2_ferromag = 0.001
N_ferromag1 = 400
N_ferromag2 = 1000

J1_ferrimag = J1_ferromag
J2_ferrimag = -0.001
N_ferrimag1 = 200
N_ferrimag2 = 500

fspm = 0.15
fferro1 = 0.1
fferro2 = 0.03
fferri1 = 0.6
fferri2 = 1-(fspm + fferro1 + fferro2 + fferri1)


############################## 	DATA ACQUISITION from files: 	####################################
num_skipped = 1  # no. of rows skipped from top
H_spm, M_spm = np.loadtxt(filepath+'./results/data_'+file_name[0]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_superparamag)+'_J1_2'+str(J1_superparamag)+'_J2_2'+str(J2_superparamag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

H_ferro1, M_ferro1 = np.loadtxt(filepath+'./results/data_'+file_name[1]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferromag1)+'_J1_2'+str(J1_ferromag)+'_J2_2'+str(J2_ferromag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
H_ferro2, M_ferro2 = np.loadtxt('./results/data_'+file_name[1]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferromag2)+'_J1_2'+str(J1_ferromag)+'_J2_2'+str(J2_ferromag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

H_ferri1, M_ferri1 = np.loadtxt(filepath+'results/data_'+file_name[2]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferrimag1)+'_J1_2'+str(J1_ferrimag)+'_J2_2'+str(J2_ferrimag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
H_ferri2, M_ferri2 = np.loadtxt(filepath+'results/data_'+file_name[2]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferrimag2)+'_J1_2'+str(J1_ferrimag)+'_J2_2'+str(J2_ferrimag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

################################ 	   M-H data         ###############################
#x, y = np.loadtxt(filepath_MH, dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2])

################################  	(DELTA)M-H data	    ###############################
B, delM = np.loadtxt(filepath_deltaMH[2], dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 1])
## convert B to H:
H = (B/mu_0) - delM   # units of H are Am-1   1 Am-1 = 0.01256637061436 Oe
#Hc = (Bc/mu_0) - Mc

#################### PLOTTING FROM DATA ACQUIRED from files: ##########################

divisor_spm = ((max(M_spm)-min(M_spm))*(mu_B*N_Avo/N_superparamag))/(max(delM)-min(delM))
divisor_ferro1 = ((max(M_ferro1)-min(M_ferro1))*(mu_B*N_Avo/N_ferromag1))/(max(delM)-min(delM))
divisor_ferro2 = ((max(M_ferro2)-min(M_ferro2))*(mu_B*N_Avo/N_ferromag2))/(max(delM)-min(delM))
divisor_ferri1 = ((max(M_ferri1)-min(M_ferri1))*(mu_B*N_Avo/N_ferrimag1))/(max(delM)-min(delM))
divisor_ferri2 = ((max(M_ferri2)-min(M_ferri2))*(mu_B*N_Avo/N_ferrimag2))/(max(delM)-min(delM))
'''
divisor_spm = 1
divisor_ferro1 = 1
divisor_ferro2 = 1
divisor_ferri1 = 1
divisor_ferri2 = 1
'''

ax[1,0].plot(H, delM, '-', color='black', label='BVDGC exp at 5K')
ax[1,0].plot(10e5*H_ferri2, (fspm*M_spm*mu_B*N_Avo/N_superparamag)/divisor_spm+(fferro1*M_ferro1*mu_B*N_Avo/N_ferromag1)/divisor_ferro1+(fferro2*M_ferro2*mu_B*N_Avo/N_ferromag2)/divisor_ferro2+(fferri1*M_ferri1*mu_B*N_Avo/N_ferrimag1)/divisor_ferri1+(fferri2*M_ferri2*mu_B*N_Avo/N_ferrimag2)/divisor_ferri2, '-', color='red', label='fit')
#plt.plot(10e5*Ha, ((fa*Ma*mu_B*N_Avo/(N_superparamag*N_superparamag))+(fb*Mb*mu_B*N_Avo/(N_ferromag*N_ferromag))+(fc*Mc*mu_B*N_Avo/(N_ferrimag*N_ferrimag)))/divisor, '-', color='navy', label='sum of signals')
########################################################################################################################














########################################################################################################################

######## best paramaters for aBVDGC (ferro+ferri):
J1_superparamag = 0.0
J2_superparamag = 0.0
N_superparamag = 80

J1_ferromag = 0.2
J2_ferromag = 0.001
N_ferromag1 = 400
N_ferromag2 = 1000

J1_ferrimag = J1_ferromag
J2_ferrimag = -0.001
N_ferrimag1 = 200
N_ferrimag2 = 500

fspm = 0.0
fferro1 = 0.01
fferro2 = 0.01
fferri1 = 0.8
fferri2 = 1-(fspm + fferro1 + fferro2 + fferri1)

########################################################################################################################
############################## 	DATA ACQUISITION from files: 	####################################
num_skipped = 1  # no. of rows skipped from top
H_spm, M_spm = np.loadtxt(filepath+'./results/data_'+file_name[0]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_superparamag)+'_J1_2'+str(J1_superparamag)+'_J2_2'+str(J2_superparamag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

H_ferro1, M_ferro1 = np.loadtxt(filepath+'./results/data_'+file_name[1]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferromag1)+'_J1_2'+str(J1_ferromag)+'_J2_2'+str(J2_ferromag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
H_ferro2, M_ferro2 = np.loadtxt('./results/data_'+file_name[1]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferromag2)+'_J1_2'+str(J1_ferromag)+'_J2_2'+str(J2_ferromag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

H_ferri1, M_ferri1 = np.loadtxt(filepath+'results/data_'+file_name[2]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferrimag1)+'_J1_2'+str(J1_ferrimag)+'_J2_2'+str(J2_ferrimag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
H_ferri2, M_ferri2 = np.loadtxt(filepath+'results/data_'+file_name[2]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferrimag2)+'_J1_2'+str(J1_ferrimag)+'_J2_2'+str(J2_ferrimag)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])







################################ 	   M-H data         ###############################
#x, y = np.loadtxt(filepath_MH, dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2])

################################  	(DELTA)M-H data	    ###############################
B, delM = np.loadtxt(filepath_deltaMH[3], dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 1])
## convert B to H:
H = (B/mu_0) - delM   # units of H are Am-1   1 Am-1 = 0.01256637061436 Oe
#Hc = (Bc/mu_0) - Mc














#################### PLOTTING FROM DATA ACQUIRED from files: ##########################

divisor_spm = ((max(M_spm)-min(M_spm))*(mu_B*N_Avo/N_superparamag))/(max(delM)-min(delM))
divisor_ferro1 = ((max(M_ferro1)-min(M_ferro1))*(mu_B*N_Avo/N_ferromag1))/(max(delM)-min(delM))
divisor_ferro2 = ((max(M_ferro2)-min(M_ferro2))*(mu_B*N_Avo/N_ferromag2))/(max(delM)-min(delM))
divisor_ferri1 = ((max(M_ferri1)-min(M_ferri1))*(mu_B*N_Avo/N_ferrimag1))/(max(delM)-min(delM))
divisor_ferri2 = ((max(M_ferri2)-min(M_ferri2))*(mu_B*N_Avo/N_ferrimag2))/(max(delM)-min(delM))
'''
divisor_spm = 1
divisor_ferro1 = 1
divisor_ferro2 = 1
divisor_ferri1 = 1
divisor_ferri2 = 1
'''

ax[1,1].plot(H, delM, '-', color='black', label='aBVDGC exp at 5K')
ax[1,1].plot(10e5*H_ferri2, (fspm*M_spm*mu_B*N_Avo/N_superparamag)/divisor_spm+(fferro1*M_ferro1*mu_B*N_Avo/N_ferromag1)/divisor_ferro1+(fferro2*M_ferro2*mu_B*N_Avo/N_ferromag2)/divisor_ferro2+(fferri1*M_ferri1*mu_B*N_Avo/N_ferrimag1)/divisor_ferri1+(fferri2*M_ferri2*mu_B*N_Avo/N_ferrimag2)/divisor_ferri2, '-', color='red', label='fit')
########################################################################################################################
















########################################################################################################################
#ax[0,0].set_xlim([xlim_low, xlim_high])
ax[0,0].set_xlabel(r"$H$/T")
ax[0,0].set_ylabel(r"$M$/emu g$^{-1}}$.")
#ax[1,0].set_ylim(0, 0.06)
#ax[0].set_ylim(bottom=0)
ax[0,0].legend()
ax[0,0].text(2.4e6, 0.0085, "(a)", fontsize=14)
#ax[1,0].set_xlim([T_low/T_divisor, T_high/T_divisor])
#ax[1,0].set_ylim(bottom=0)
#ax[1,0].set_ylim(0,2.5)
ax[0,1].set_xlabel(r"$H$/T")
ax[0,1].set_ylabel(r"$M$/emu g$^{-1}}$.")
ax[0,1].legend()
ax[0,1].text(2.4e6, 0.0038, "(b)", fontsize=14)

ax[1,0].set_xlabel(r"$H$/T")
ax[1,0].set_ylabel(r"$M$/emu g$^{-1}}$.")
ax[1,0].legend()
ax[1,0].text(2.4e6, 0.012, "(c)", fontsize=14)

ax[1,1].set_xlabel(r"$H$/T")
ax[1,1].set_ylabel(r"$M$/emu g$^{-1}}$.")
ax[1,1].legend()
ax[1,1].text(2.4e6, 0.05, "(d)", fontsize=14)

#ax[0,0].text(0, -0.02, [J1_2, J2_2, N_2, divisor], fontsize=12)

########################################################################################################################
plt.show()
fig.savefig(filepath+"results/Final_Plot"+"_Nitt"+str(Nitt)+"_"+str(warm)+"_"+str(measure)+"_"+".png", format="png",dpi=600)
plt.close()
