#!/usr/bin/env python
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
from parameters_MT import *	# parameter definitions are stored in this file
from func_mag import *		# function definitions are stored in this file
import matplotlib.pyplot as plt
'''
print(str(sys.argv))
print(sys.argv[0]) # prints python_script.py
print(sys.argv[1]) # prints var2

J3 = float(sys.argv[1])

if len(sys.argv) > 2:
	J2_2 = float(sys.argv[2])
else:
	J2_2 = sys_params[6]

print("J3 = ", str(J3), "J1_2 = ", str(J1_2), "J2_2 = ", str(J2_2), "Nitt = "+str(Nitt), "N = "+str(N_2))
'''

# USING ARGUMENT PARSER (argparse package) to use arguments (in case of more than 2 arguments passed from terminal to script):
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--N', nargs = 1)
parser.add_argument('--J1', nargs = 1)
parser.add_argument('--J2', nargs = 1)
parser.add_argument('--J3', nargs = 1)
args = parser.parse_args()
#print(args)
N_2 = int(args.N[0])
J1_2 = float(args.J1[0])
J2_2 = float(args.J2[0])
J3 = float(args.J3[0])
#J3 = float(args.J3[0])


if J2_2 < 0:
	FERRO = False
elif J2_2 > 0:
	FERRO = True	
FERRI = 1-FERRO

print(" ")
print("Start of Calculation, [J1_2, J2_2, J3, N_2] = ", [J1_2, J2_2, J3, N_2])


#########################################################################################################################
if J1_2 == 0 and J2_2 == 0:
	system_name = 'superparamag'
	Mag_label = r'M$_{superparamag}$'
	plot_color = ['black', 'red']
	chi_label = 'superparamag chi'
elif FERRO == True:
	system_name = 'ferromag'
	Mag_label = r'M$_{ferromag}$'
	plot_color = ['blue', 'green']
	chi_label = 'ferromag chi'
elif FERRI == True:
	system_name = 'ferrimag'
	Mag_label = r'M$_{ferrimag}$'
	plot_color = ['brown', 'navy']
	chi_label = 'ferrimag chi'
	
#######	Execute commands for Calculations:  #############################################################################

if path.exists("./results/data_"+system_name+"_J3"+str(J3)+"_Nitt"+str(Nitt)+"_T_high"+str(T_high)+"_warmsteps"+str(warm)+"_measure"+str(measure)+"_N_2"+str(N_2)+"_J1_2"+str(J1_2)+"_J2_2"+str(J2_2)+".csv") == False:
	(edge1, edge2) = RandomL(N_2)
	#edge1 = ones((N), dtype=int)
	#edge2 = ones((N), dtype=int)
#	(edge1, edge2) = Equilibriate(warm, edge1, edge2, J1_2, J2_2, J3, T_high, T_divisor)
	(wMag, wEne, wTemp, wChi) = MT_loop(Nitt, edge1, edge2, J1_2, J2_2, J3, T_high, T_low, T_int, T_divisor)
	f = open("./results/data_"+system_name+"_J3"+str(J3)+"_Nitt"+str(Nitt)+"_T_high"+str(T_high/T_divisor)+"_warmsteps"+str(warm)+"_measure"+str(measure)+"_N_2"+str(N_2)+"_J1_2"+str(J1_2)+"_J2_2"+str(J2_2)+".csv", "w")
	f.write("{},{},{},{}\n".format("Temp", "Energy", "Magnetization", "Susceptibility"))
	for x in zip(wTemp, wEne, wMag, wChi):
		f.write("{},{},{},{}\n".format(x[0], x[1], x[2], x[3]))
	f.close()



#########################################################################################################
######################################### PLOTTING COMMANDS #############################################
#########################################################################################################
#########################################################################################################

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



#######################################################################
##########################      PLOTS:		 ##########################
#######################################################################

x, y, chi = np.loadtxt('./results/data_'+system_name+'_J3'+str(J3)+'_Nitt'+str(Nitt)+'_T_high'+str(T_high/T_divisor)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_2)+'_J1_2'+str(J1_2)+'_J2_2'+str(J2_2)+'.csv', dtype='double', delimiter=',', skiprows=num_skipped, unpack=True, usecols=[0, 2, 3])

fig, ax = plt.subplots(2,1)

ax[0].plot(x, y*mu_B*N_Avo, '-', color=plot_color[0], label=Mag_label)
#if J3 == 0:
#	ax[1].plot(x, chi/N_Avo, '-', color=plot_color[1], label= chi_label)
#elif J3 > 0:
#	ax[1].plot(x, (chi-chi[len(chi)-1])/N_Avo, '-', color=plot_color[1], label= chi_label)  # Phys. Chem. Chem. Phys., 16 (2014) 6273-6282
ax[1].plot(x, chi/N_Avo, '-', color=plot_color[1], label= chi_label)  # Phys. Chem. Chem. Phys., 16 (2014) 6273-6282
ax[0].set_xlim([xlim_low, xlim_high])
ax[0].set_xlabel(r"$T$/K")
ax[0].set_ylabel(r"$M$/emu g$^{-1}}$.")
#ax[0].set_ylim(bottom=0)
#ax[0].set_ylim([ylim_low, ylim_high])
ax[1].set_xlim([T_low/T_divisor, T_high/T_divisor])
ax[1].set_xlabel(r"$T$/K")
ax[1].set_ylabel(r"$\chi$")
ax[1].set_ylim(bottom=0)
#ax[1].set_ylim([0, 1])
fig.savefig("./results/plot_"+system_name+"_J3"+str(J3)+"_Nitt"+str(Nitt)+"_T_high"+str(T_high/T_divisor)+"_warmsteps"+str(warm)+"_measure"+str(measure)+"_N_2"+str(N_2)+"_J1_2"+str(J1_2)+"_J2_2"+str(J2_2)+".png", format="png",dpi=600)
#plt.show()

plt.close()


print('data_'+system_name+'_J3'+str(J3)+'_Nitt'+str(Nitt)+'_T_high'+str(T_high/T_divisor)+'_warmsteps'+str(warm)+'_measure'+str(measure)+'_N_2'+str(N_2)+'_J1_2'+str(J1_2)+'_J2_2'+str(J2_2)+'.csv')















####################################       EXTRA CODE        ###############################################
'''
print(str(sys.argv))
print(sys.argv[0]) # prints python_script.py
print(sys.argv[1]) # prints var2
print(sys.argv[2]) # prints var2

J1_2 = float(sys.argv[1])
J2_2 = float(sys.argv[2])
print("J1_2 = ", str(J1_2), "J2_2 = ", str(J2_2), "Nitt = "+str(Nitt), "N = "+str(N_2))


print("Started Calculation")
'''


'''
#################  TEMPERATURE and ARRAY SIZE input commands: ##############################

N = int(input('N_superparamag = '))	#10      # linear dimension of the 1st (Superparamag) lattice, lattice-size= N x N
N_2 = int(input('N_ferromag = ')) #2000        # linear dimension of the 2nd (Ferromag) lattice components, lattice-size= N_2 x N_2
T = int(input('T = '))
'''
		#print([aE, aM])

'''
file = open("data_superparamag_Nitt"+str(Nitt)+"_J3"+str(J3)+"_N"+str(N)+"_J1"+str(J1)+"_J2"+str(J2)+".csv", "r")
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

