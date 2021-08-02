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
'
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

Nitt = 50000  				# total number of Monte Carlo steps
warm = 1000    			# Number of warmup steps
measure=100     			# How often to take a measurement
T_max = 10					# T_max*T_divisor = T_high (the integral input to functions, divided inside function again by T_divisor) maximum temperature of system in kelvins
T_divisor=10
#T_max = 300					# maximum temperature of system in kelvins
#T_divisor=1
############################ PHYSICAL CONSTANTS: ###################################################################
k_B = 1.381e-23				# Boltzmann constant (Joules/Kelvin)
mu_o = 1.25663706212e-6		# magnetic permeability of vacuum (H/m)
mu_B = 9.274e-24 			# Bohr magneton (Joules/Tesla)
g = 2.002 					# spin g-factor for free electrons (for ferromag electrons)
h_bar = 1.054e-34			# reduced Planck's constant
m_e = 9.109e-31				# electron rest mass (kg)
N_Avo = 6.022e23			# Avogadro number

'''
System parameters are set as:
sys_params = [Nitt, warming, measure, T_max, N, J1, J2]
'''
 
'''
parameters for ferrimagnet case:
'''
#sys_params = [Nitt, warm, measure, T_max, 200, 0.4, -0.001]
#sys_params = [Nitt, warm, measure, T_max, 200, 0.4, -0.002]

'''
parameters for ferromagnet case:
'''
sys_params = [Nitt, warm, measure, T_max, 200, 0.2, 0.001]
#sys_params = [Nitt, warm, measure, T_max, 200, 0.3, 0.003]
'''
parameters for superparamagnet case:
'''
#sys_params = [Nitt, warm, measure, T_max, 50, 1.7, 0.0]

#########################################################################################################################
###########################################                           ###################################################
########################################### PARAMETERS FOR M-T plots: ###################################################
###########################################                           ###################################################
#########################################################################################################################

#################  EXTERNAL FIELD and NANOCARBON EDGE SIZE: #########################################################
#J3 = 1					# J3 = 0 for Zero field cooling (ZFC), J3 != 0 for Field cooling (FC) curves
N_2 = sys_params[4]	      		# linear dimension of the 2nd (Ferromag) lattice components, lattice-size= N_2 x N_2
						# N_2 = 2000, J1_1 = 1, and J2_0 = 0 give good ferromagnetic plot
if N_2 > 80:
	flag_spm = False	#particles are NOT superparamagnetic if flag_spm = False
elif N_2 <= 80:
	flag_spm = True		# particles are superparamagnetic if flag_spm = True
#N = 20		  	   		# linear dimension of the 1st (Superparamag) lattice, lattice-size= N x N

######### parameters for SUPERPARAMAGNETIC RUN (N != 0, N2 != 0, FERRO = True): ################
#J1 = 1.7
#J2 = 0.0
######### parameters for FERRO-/FERRIMAGNETIC RUN: ##################################################################
###### for FERROMAG RUN: (N == 0, N2 != 0, FERRO = True) ############################################################
###### for FERRIMAG RUN: (N == 0, N2 != 0, FERRO = False) ###########################################################
J1_2 = sys_params[5]
J2_2 = sys_params[6]			# getting good fitting (for aBVLGC) for J1 = 1, J2 = -0.05 and T = 5K, N has to be kept 50 or above

if J2_2 < 0:
	FERRO = False
else:
	FERRO = True	
FERRI = 1-FERRO

######################################################################
T_high = sys_params[3]*T_divisor
T_low = 0				# Temperature range is [T_high, T_low]
T_int = 1				# to make data more resolved, we divide the interval of the list/array T by this number, 10, 100, or so.
######################################################################
#f1 = 1				# fraction of signal from each measurement to be added to get final signal
#f2 = 0
#f3 = 1-f1-f2
#divisor = 5				# factor to divide final graph values to obtain perfect fit with good y-axis match of experimental plot (because not all of the material in the real/synthesized carbon is graphitic/graphenic. Infact, this factor gives and idea about the fraction of material that is graphenic  (in the current case, 30May2021, BVDGC, divisor = 4)
########################################################################
num_skipped = 1  		# no. of rows skipped from top
##################### range of axes: ###################################
xlim_low = T_low/T_divisor
xlim_high = T_high/T_divisor
ylim_low = -1
ylim_high = 1
