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
from parameters_MT import *
import sys

#J3 = float(sys.argv[1])

# BVDGC is SPM+Ferro+Ferri
# aBVDGC is pure Ferro + Ferri


colorstore = ['black','red','blue','green','orange','yellow','pink','navy','magenta','cyan','brown','gray','purple']


#fig, ax = plt.subplots(1,3)

'''
axins1 = inset_axes(ax[1,1], width="100%", height="100%",
			bbox_to_anchor=(.2, .2, .5, .6),
			bbox_transform=ax[1,0].transAxes, loc=3)
axins2 = inset_axes(ax[1,0], width="100%", height="100%",
			bbox_to_anchor=(.2, .2, .5, .6),
			bbox_transform=ax[1,1].transAxes, loc=3)
axins3 = inset_axes(ax[0,0], width="100%", height="100%",
			bbox_to_anchor=(.2, .3, .5, .6),
			bbox_transform=ax[0,0].transAxes, loc=3)
'''


########################################################################

path = '/home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-T_MC'

########################################################################
filepath_deltaMH=['/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_petal/PPMS_BVPGC/MH_BVPGC_Diam_subt.csv', '/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_petal/PPMS_aBVPGC/MH_aBVPGC_Diam_subt.csv', '/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_detritus/PPMS_BVDGC/MH_BVDGC_Diam_subt.csv', '/home/nitish/Desktop/JNU/Work_/My_manuscripts/Analysis/Magnetization/MH_detritus/PPMS_a-BVDGC/MH_aBVDGC_Diam_subt.csv']
filepath = '/home/nitish/Desktop/Ising/My_Ising_New_15062021/Detritus_M-H_MC/'
file_name = ['superparamag', 'ferromag', 'ferrimag']

num_skipped = 1  # no. of rows skipped from top


# BVDGC is SPM+Ferro+Ferri
# aBVDGC is pure Ferro + Ferri
T_high = 10.0



fig, ax = plt.subplots(2,2)













######## best paramaters for BVPGC (spm+ferro+ferri):

J1_spm = 0.0
J2_spm = 0.0
N_spm = 80

J1_ferro = 0.2
J2_ferro = 0.001
N_ferro1 = 400
N_ferro2 = 1000

J1_ferri = J1_ferro
J2_ferri = -0.001
N_ferri1 = 200
N_ferri2 = 500

fspm = 0.02
fferro1 = 0.05
fferro2 = 0.05
fferri1 = 0.32
fferri2 = 1-(fspm + fferro1 + fferro2 + fferri1)


filepathindex = 0
##################################################
H_spm, M_spm = np.loadtxt(filepath+'./results/data_'+file_name[0]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_spm)+'_J1_2'+str(J1_spm)+'_J2_2'+str(J2_spm)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

H_ferro1, M_ferro1 = np.loadtxt(filepath+'./results/data_'+file_name[1]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferro1)+'_J1_2'+str(J1_ferro)+'_J2_2'+str(J2_ferro)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
H_ferro2, M_ferro2 = np.loadtxt(filepath+'./results/data_'+file_name[1]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferro2)+'_J1_2'+str(J1_ferro)+'_J2_2'+str(J2_ferro)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

H_ferri1, M_ferri1 = np.loadtxt(filepath+'results/data_'+file_name[2]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferri1)+'_J1_2'+str(J1_ferri)+'_J2_2'+str(J2_ferri)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
H_ferri2, M_ferri2 = np.loadtxt(filepath+'results/data_'+file_name[2]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferri2)+'_J1_2'+str(J1_ferri)+'_J2_2'+str(J2_ferri)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])


B, delM = np.loadtxt(filepath_deltaMH[0], dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 1])

mu_0 = 4*math.pi*(10**(-7))

divisor_spm = ((max(M_spm)-min(M_spm))*(mu_B*N_Avo/N_spm))/(max(delM)-min(delM))
divisor_ferro1 = ((max(M_ferro1)-min(M_ferro1))*(mu_B*N_Avo/N_ferro1))/(max(delM)-min(delM))
divisor_ferro2 = ((max(M_ferro2)-min(M_ferro2))*(mu_B*N_Avo/N_ferro2))/(max(delM)-min(delM))
divisor_ferri1 = ((max(M_ferri1)-min(M_ferri1))*(mu_B*N_Avo/N_ferri1))/(max(delM)-min(delM))
divisor_ferri2 = ((max(M_ferri2)-min(M_ferri2))*(mu_B*N_Avo/N_ferri2))/(max(delM)-min(delM))

J3 = []
T_plot_spm = []
M_plot_spm = []
chi_plot_spm = []
M_plot_ferro1 = []
chi_plot_ferro1 = []
M_plot_ferro2 = []
chi_plot_ferro2 = []
M_plot_ferri1 = []
chi_plot_ferri1 = []
M_plot_ferri2 = []
chi_plot_ferri2 = []

for i in range(0, 2, 1):
	J3.append(float(i/2))	
	xspm, yspm, zspm = np.loadtxt(path+'/results/data_superparamag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_spm)+'_J1_2'+str(J1_spm)+'_J2_2'+str(J2_spm)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	xferro1, yferro1, zferro1 = np.loadtxt(path+'/results/data_ferromag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferro1)+'_J1_2'+str(J1_ferro)+'_J2_2'+str(J2_ferro)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	xferro2, yferro2, zferro2 = np.loadtxt(path+'/results/data_ferromag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferro2)+'_J1_2'+str(J1_ferro)+'_J2_2'+str(J2_ferro)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	xferri1, yferri1, zferri1 = np.loadtxt(path+'/results/data_ferrimag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferri1)+'_J1_2'+str(J1_ferri)+'_J2_2'+str(J2_ferri)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	xferri2, yferri2, zferri2 = np.loadtxt(path+'/results/data_ferrimag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferri2)+'_J1_2'+str(J1_ferri)+'_J2_2'+str(J2_ferri)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	T_plot_spm.append(xspm)
	M_plot_spm.append(yspm)
	M_plot_ferro1.append(yferro1)
	M_plot_ferro2.append(yferro2)
	M_plot_ferri1.append(yferri1)
	M_plot_ferri2.append(yferri2)
	
	ax[0,0].plot(T_plot_spm[i], ((fspm*M_plot_spm[i]*mu_B*N_Avo/(N_spm*N_spm))/divisor_spm + (fferro1*M_plot_ferro1[i]*mu_B*N_Avo/(N_ferro1*N_ferro1))/divisor_ferro1 + (fferro2*M_plot_ferro2[i]*mu_B*N_Avo/(N_ferro2*N_ferro2))/divisor_ferro2 + (fferri1*M_plot_ferri1[i]*mu_B*N_Avo/(N_ferri1*N_ferri1))/divisor_ferri1 +(fferri2*M_plot_ferri2[i]*mu_B*N_Avo/(N_ferri2*N_ferri2))/divisor_ferri2), '-', color=colorstore[i], label=str(J3[len(J3)-1]))

########################################################################################################################














######## best paramaters for a-BVPGC (spm+ferro+ferri):

J1_spm = 0.0
J2_spm = 0.0
N_spm = 80

J1_ferro = 0.2
J2_ferro = 0.001
N_ferro1 = 400
N_ferro2 = 1000

J1_ferri = J1_ferro
J2_ferri = -0.001
N_ferri1 = 200
N_ferri2 = 500

fspm = 0.02
fferro1 = 0.05
fferro2 = 0.05
fferri1 = 0.32
fferri2 = 1-(fspm + fferro1 + fferro2 + fferri1)


filepathindex = 0
##################################################
H_spm, M_spm = np.loadtxt(filepath+'./results/data_'+file_name[0]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_spm)+'_J1_2'+str(J1_spm)+'_J2_2'+str(J2_spm)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

H_ferro1, M_ferro1 = np.loadtxt(filepath+'./results/data_'+file_name[1]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferro1)+'_J1_2'+str(J1_ferro)+'_J2_2'+str(J2_ferro)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
H_ferro2, M_ferro2 = np.loadtxt(filepath+'./results/data_'+file_name[1]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferro2)+'_J1_2'+str(J1_ferro)+'_J2_2'+str(J2_ferro)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

H_ferri1, M_ferri1 = np.loadtxt(filepath+'results/data_'+file_name[2]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferri1)+'_J1_2'+str(J1_ferri)+'_J2_2'+str(J2_ferri)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
H_ferri2, M_ferri2 = np.loadtxt(filepath+'results/data_'+file_name[2]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferri2)+'_J1_2'+str(J1_ferri)+'_J2_2'+str(J2_ferri)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])


B, delM = np.loadtxt(filepath_deltaMH[1], dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 1])

mu_0 = 4*math.pi*(10**(-7))

divisor_spm = ((max(M_spm)-min(M_spm))*(mu_B*N_Avo/N_spm))/(max(delM)-min(delM))
divisor_ferro1 = ((max(M_ferro1)-min(M_ferro1))*(mu_B*N_Avo/N_ferro1))/(max(delM)-min(delM))
divisor_ferro2 = ((max(M_ferro2)-min(M_ferro2))*(mu_B*N_Avo/N_ferro2))/(max(delM)-min(delM))
divisor_ferri1 = ((max(M_ferri1)-min(M_ferri1))*(mu_B*N_Avo/N_ferri1))/(max(delM)-min(delM))
divisor_ferri2 = ((max(M_ferri2)-min(M_ferri2))*(mu_B*N_Avo/N_ferri2))/(max(delM)-min(delM))

J3 = []
T_plot_spm = []
M_plot_spm = []
chi_plot_spm = []
M_plot_ferro1 = []
chi_plot_ferro1 = []
M_plot_ferro2 = []
chi_plot_ferro2 = []
M_plot_ferri1 = []
chi_plot_ferri1 = []
M_plot_ferri2 = []
chi_plot_ferri2 = []

for i in range(0, 2, 1):
	J3.append(float(i/2))	
	xspm, yspm, zspm = np.loadtxt(path+'/results/data_superparamag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_spm)+'_J1_2'+str(J1_spm)+'_J2_2'+str(J2_spm)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	xferro1, yferro1, zferro1 = np.loadtxt(path+'/results/data_ferromag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferro1)+'_J1_2'+str(J1_ferro)+'_J2_2'+str(J2_ferro)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	xferro2, yferro2, zferro2 = np.loadtxt(path+'/results/data_ferromag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferro2)+'_J1_2'+str(J1_ferro)+'_J2_2'+str(J2_ferro)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	xferri1, yferri1, zferri1 = np.loadtxt(path+'/results/data_ferrimag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferri1)+'_J1_2'+str(J1_ferri)+'_J2_2'+str(J2_ferri)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	xferri2, yferri2, zferri2 = np.loadtxt(path+'/results/data_ferrimag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferri2)+'_J1_2'+str(J1_ferri)+'_J2_2'+str(J2_ferri)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	T_plot_spm.append(xspm)
	M_plot_spm.append(yspm)
	M_plot_ferro1.append(yferro1)
	M_plot_ferro2.append(yferro2)
	M_plot_ferri1.append(yferri1)
	M_plot_ferri2.append(yferri2)
	
	ax[0,1].plot(T_plot_spm[i], ((fspm*M_plot_spm[i]*mu_B*N_Avo/(N_spm*N_spm))/divisor_spm + (fferro1*M_plot_ferro1[i]*mu_B*N_Avo/(N_ferro1*N_ferro1))/divisor_ferro1 + (fferro2*M_plot_ferro2[i]*mu_B*N_Avo/(N_ferro2*N_ferro2))/divisor_ferro2 + (fferri1*M_plot_ferri1[i]*mu_B*N_Avo/(N_ferri1*N_ferri1))/divisor_ferri1 +(fferri2*M_plot_ferri2[i]*mu_B*N_Avo/(N_ferri2*N_ferri2))/divisor_ferri2), '-', color=colorstore[i], label=str(J3[len(J3)-1]))

########################################################################################################################












######## best paramaters for BVDGC (spm+ferro+ferri):

J1_spm = 0.0
J2_spm = 0.0
N_spm = 80

J1_ferro = 0.2
J2_ferro = 0.001
N_ferro1 = 400
N_ferro2 = 1000

J1_ferri = J1_ferro
J2_ferri = -0.001
N_ferri1 = 200
N_ferri2 = 500

fspm = 0.02
fferro1 = 0.05
fferro2 = 0.05
fferri1 = 0.32
fferri2 = 1-(fspm + fferro1 + fferro2 + fferri1)


filepathindex = 0
##################################################
H_spm, M_spm = np.loadtxt(filepath+'./results/data_'+file_name[0]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_spm)+'_J1_2'+str(J1_spm)+'_J2_2'+str(J2_spm)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

H_ferro1, M_ferro1 = np.loadtxt(filepath+'./results/data_'+file_name[1]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferro1)+'_J1_2'+str(J1_ferro)+'_J2_2'+str(J2_ferro)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
H_ferro2, M_ferro2 = np.loadtxt(filepath+'./results/data_'+file_name[1]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferro2)+'_J1_2'+str(J1_ferro)+'_J2_2'+str(J2_ferro)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

H_ferri1, M_ferri1 = np.loadtxt(filepath+'results/data_'+file_name[2]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferri1)+'_J1_2'+str(J1_ferri)+'_J2_2'+str(J2_ferri)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
H_ferri2, M_ferri2 = np.loadtxt(filepath+'results/data_'+file_name[2]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferri2)+'_J1_2'+str(J1_ferri)+'_J2_2'+str(J2_ferri)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])


B, delM = np.loadtxt(filepath_deltaMH[2], dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 1])

mu_0 = 4*math.pi*(10**(-7))

divisor_spm = ((max(M_spm)-min(M_spm))*(mu_B*N_Avo/N_spm))/(max(delM)-min(delM))
divisor_ferro1 = ((max(M_ferro1)-min(M_ferro1))*(mu_B*N_Avo/N_ferro1))/(max(delM)-min(delM))
divisor_ferro2 = ((max(M_ferro2)-min(M_ferro2))*(mu_B*N_Avo/N_ferro2))/(max(delM)-min(delM))
divisor_ferri1 = ((max(M_ferri1)-min(M_ferri1))*(mu_B*N_Avo/N_ferri1))/(max(delM)-min(delM))
divisor_ferri2 = ((max(M_ferri2)-min(M_ferri2))*(mu_B*N_Avo/N_ferri2))/(max(delM)-min(delM))

J3 = []
T_plot_spm = []
M_plot_spm = []
chi_plot_spm = []
M_plot_ferro1 = []
chi_plot_ferro1 = []
M_plot_ferro2 = []
chi_plot_ferro2 = []
M_plot_ferri1 = []
chi_plot_ferri1 = []
M_plot_ferri2 = []
chi_plot_ferri2 = []

for i in range(0, 2, 1):
	J3.append(float(i/2))	
	xspm, yspm, zspm = np.loadtxt(path+'/results/data_superparamag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_spm)+'_J1_2'+str(J1_spm)+'_J2_2'+str(J2_spm)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	xferro1, yferro1, zferro1 = np.loadtxt(path+'/results/data_ferromag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferro1)+'_J1_2'+str(J1_ferro)+'_J2_2'+str(J2_ferro)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	xferro2, yferro2, zferro2 = np.loadtxt(path+'/results/data_ferromag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferro2)+'_J1_2'+str(J1_ferro)+'_J2_2'+str(J2_ferro)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	xferri1, yferri1, zferri1 = np.loadtxt(path+'/results/data_ferrimag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferri1)+'_J1_2'+str(J1_ferri)+'_J2_2'+str(J2_ferri)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	xferri2, yferri2, zferri2 = np.loadtxt(path+'/results/data_ferrimag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferri2)+'_J1_2'+str(J1_ferri)+'_J2_2'+str(J2_ferri)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	T_plot_spm.append(xspm)
	M_plot_spm.append(yspm)
	M_plot_ferro1.append(yferro1)
	M_plot_ferro2.append(yferro2)
	M_plot_ferri1.append(yferri1)
	M_plot_ferri2.append(yferri2)
	
	ax[1,0].plot(T_plot_spm[i], ((fspm*M_plot_spm[i]*mu_B*N_Avo/(N_spm*N_spm))/divisor_spm + (fferro1*M_plot_ferro1[i]*mu_B*N_Avo/(N_ferro1*N_ferro1))/divisor_ferro1 + (fferro2*M_plot_ferro2[i]*mu_B*N_Avo/(N_ferro2*N_ferro2))/divisor_ferro2 + (fferri1*M_plot_ferri1[i]*mu_B*N_Avo/(N_ferri1*N_ferri1))/divisor_ferri1 +(fferri2*M_plot_ferri2[i]*mu_B*N_Avo/(N_ferri2*N_ferri2))/divisor_ferri2), '-', color=colorstore[i], label=str(J3[len(J3)-1]))

########################################################################################################################
























######## best paramaters for aBVDGC (ferro+ferri):
J1_spm = 0.0
J2_spm = 0.0
N_spm = 80

J1_ferro = 0.2
J2_ferro = 0.001
N_ferro1 = 400
N_ferro2 = 1000

J1_ferri = J1_ferro
J2_ferri = -0.001
N_ferri1 = 200
N_ferri2 = 500

fspm = 0.0
fferro1 = 0.00
fferro2 = 0.001
fferri1 = 0.998
fferri2 = 1-(fspm + fferro1 + fferro2 + fferri1)

##################################################
H_spm, M_spm = np.loadtxt(filepath+'./results/data_'+file_name[0]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_spm)+'_J1_2'+str(J1_spm)+'_J2_2'+str(J2_spm)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

H_ferro1, M_ferro1 = np.loadtxt(filepath+'./results/data_'+file_name[1]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferro1)+'_J1_2'+str(J1_ferro)+'_J2_2'+str(J2_ferro)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
H_ferro2, M_ferro2 = np.loadtxt(filepath+'./results/data_'+file_name[1]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferro2)+'_J1_2'+str(J1_ferro)+'_J2_2'+str(J2_ferro)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])

H_ferri1, M_ferri1 = np.loadtxt(filepath+'results/data_'+file_name[2]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferri1)+'_J1_2'+str(J1_ferri)+'_J2_2'+str(J2_ferri)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])
H_ferri2, M_ferri2 = np.loadtxt(filepath+'results/data_'+file_name[2]+'_T'+'5'+'_Nitt'+str(Nitt)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferri2)+'_J1_2'+str(J1_ferri)+'_J2_2'+str(J2_ferri)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2])


B, delM = np.loadtxt(filepath_deltaMH[3], dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 1])

mu_0 = 4*math.pi*(10**(-7))

divisor_spm = ((max(M_spm)-min(M_spm))*(mu_B*N_Avo/N_spm))/(max(delM)-min(delM))
divisor_ferro1 = ((max(M_ferro1)-min(M_ferro1))*(mu_B*N_Avo/N_ferro1))/(max(delM)-min(delM))
divisor_ferro2 = ((max(M_ferro2)-min(M_ferro2))*(mu_B*N_Avo/N_ferro2))/(max(delM)-min(delM))
divisor_ferri1 = ((max(M_ferri1)-min(M_ferri1))*(mu_B*N_Avo/N_ferri1))/(max(delM)-min(delM))
divisor_ferri2 = ((max(M_ferri2)-min(M_ferri2))*(mu_B*N_Avo/N_ferri2))/(max(delM)-min(delM))

J3 = []
T_plot_spm = []
M_plot_spm = []
chi_plot_spm = []
M_plot_ferro1 = []
chi_plot_ferro1 = []
M_plot_ferro2 = []
chi_plot_ferro2 = []
M_plot_ferri1 = []
chi_plot_ferri1 = []
M_plot_ferri2 = []
chi_plot_ferri2 = []

for i in range(0, 2, 1):
	J3.append(float(i/2))	
	xspm, yspm, zspm = np.loadtxt(path+'/results/data_superparamag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_spm)+'_J1_2'+str(J1_spm)+'_J2_2'+str(J2_spm)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	xferro1, yferro1, zferro1 = np.loadtxt(path+'/results/data_ferromag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferro1)+'_J1_2'+str(J1_ferro)+'_J2_2'+str(J2_ferro)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	xferro2, yferro2, zferro2 = np.loadtxt(path+'/results/data_ferromag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferro2)+'_J1_2'+str(J1_ferro)+'_J2_2'+str(J2_ferro)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	xferri1, yferri1, zferri1 = np.loadtxt(path+'/results/data_ferrimag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferri1)+'_J1_2'+str(J1_ferri)+'_J2_2'+str(J2_ferri)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	xferri2, yferri2, zferri2 = np.loadtxt(path+'/results/data_ferrimag'+'_J3'+str(J3[i])+'_Nitt'+str(Nitt)+'_T_high'+str(T_high)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_ferri2)+'_J1_2'+str(J1_ferri)+'_J2_2'+str(J2_ferri)+'.csv', dtype='double', delimiter=',', skiprows=1, unpack=True, usecols=[0, 2, 3])
	T_plot_spm.append(xspm)
	M_plot_spm.append(yspm)
	M_plot_ferro1.append(yferro1)
	M_plot_ferro2.append(yferro2)
	M_plot_ferri1.append(yferri1)
	M_plot_ferri2.append(yferri2)

	ax[1,1].plot(T_plot_spm[i], ((fspm*M_plot_spm[i]*mu_B*N_Avo/(N_spm*N_spm))/divisor_spm + (fferro1*M_plot_ferro1[i]*mu_B*N_Avo/(N_ferro1*N_ferro1))/divisor_ferro1 + (fferro2*M_plot_ferro2[i]*mu_B*N_Avo/(N_ferro2*N_ferro2))/divisor_ferro2 + (fferri1*M_plot_ferri1[i]*mu_B*N_Avo/(N_ferri1*N_ferri1))/divisor_ferri1 + (fferri2*M_plot_ferri2[i]*mu_B*N_Avo/(N_ferri2*N_ferri2))/divisor_ferri2), '-', color=colorstore[i], label=str(J3[len(J3)-1]))


########################################################################################################################









#ax[2].set_ylim(bottom=0)
#ax[2].set_ylim([ylim_low, ylim_high])
#ax[0,0].set_xlim([xlim_low, xlim_high])
ax[0,0].set_xlabel(r"$T$/K")
ax[0,0].set_ylabel(r"$M$/emu g$^{-1}}$.")
#ax[0,1].set_ylim(0, 0.06)
#ax[0].set_ylim(bottom=0)
ax[0,0].legend()
#ax[1,0].set_xlim([T_low/T_divisor, T_high/T_divisor])
#ax[1,0].set_ylim(bottom=0)
#ax[1,0].set_ylim(0,2.5)
ax[1,0].set_xlabel(r"$T$/K")
ax[1,0].set_ylabel(r"$M$/emu g$^{-1}}$.")
ax[1,0].legend()

x_low, x_high, y_low, y_high = T_low, T_high, 0, 1.25 # specify the limits
#x_low, x_high, y_low, y_high = T_low/T_divisor, 50, 0, 0.05 # specify the limits
#axins1.set_xlim(x_low, x_high) # apply the x-limits
#axins1.set_ylim(y_low, y_high) # apply the y-limits
#axins3.set_xlim(0, 10) # apply the x-limits
#axins3.set_xlim(0, 10) # apply the x-limits
#axins3.set_ylim(0, 0.013) # apply the y-limits
#axins2.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')

#ax[1,1].set_xlim([T_low/T_divisor, 50])
#ax[1,1].set_ylim(bottom=0)
#ax[1,1].set_xlabel(r"$T$/K")
#ax[1,1].set_ylabel(r"$\chi$")
#ax[1,1].set_ylim([0, 1])
#ax[1,1].legend()

#ax[0].set_ylim(bottom=0)
#ax[0].set_ylim([ylim_low, ylim_high])
#ax[0,1].set_xlim([xlim_low, xlim_high])
ax[0,1].set_xlabel(r"$T$/K")
ax[0,1].set_ylabel(r"$M$/emu g$^{-1}}$.")
#ax[0,0].set_ylim(0, 0.06)
#ax[0].set_ylim(bottom=0)
ax[0,1].legend()	
#ax[1,1].set_xlim([T_low/T_divisor, T_high/T_divisor])
#ax[1,1].set_ylim(bottom=0)
#ax[1,0].set_ylim(0, 0.25)
ax[1,1].set_xlabel(r"$T$/K")
ax[1,1].set_ylabel(r"$M$/emu g$^{-1}}$.")
ax[1,1].legend()

x_low, x_high, y_low, y_high = T_low/T_divisor, T_high/T_divisor, 0, 2.6 # specify the limits
#x_low, x_high, y_low, y_high = T_low/T_divisor, 50, 1, 3 # specify the limits
#axins2.set_xlim(x_low, x_high) # apply the x-limits
#axins2.set_ylim(y_low, y_high) # apply the y-limits
#axins.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')









#plt.legend()

#fig.savefig("./results/Summed_MT_plot"+"All_J3"+"_Nitt"+str(Nitt)+"_"+str(T_max)+str(warm)+"_"+str(measure)+"_spmfr_"+str(fspm)+"fm1fr_"+str(fferro1)+"fm2fr_"+str(fferro2)+"fim1fr_"+str(fferri1)+"fim2fr_"+str(fferri2)+"_"+str(J1_spm)+"_"+str(J2_spm)+"_"+str(J1_ferro)+"_"+str(J2_ferro)+"_"+str(J1_ferri)+"_"+str(J2_ferri)+"_"+str(N_spm)+"_"+str(N_ferro1)+"_"+str(N_ferro2)+str(N_ferri1)+"_"+str(N_ferri2)+".png", format="png", bbox = 'tight', dpi=1200)
fig.savefig("./results/Summed_MT_plot.png", format="png", dpi=1200)
plt.show()
plt.close()

