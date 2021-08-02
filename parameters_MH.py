#########################################################################################################################
###########################################                           ###################################################
########################################### PARAMETERS FOR M-H plots: ###################################################
###########################################                           ###################################################
#########################################################################################################################

Nitt = 50000  				# total number of Monte Carlo steps
warm = 1000    				# Number of warmup steps
measure=100     			# How often to take a measurement
############################ physical constants: ##########################
k_B = 1.381e-23				# Boltzmann constant (Joules/Kelvin)
mu_o = 1.25663706212e-6		# magnetic permeability of vacuum (H/m)
mu_B = 9.274e-24 			# Bohr magneton (Joules/Tesla)
#g = 16*2.002 					# spin g-factor for superparamag model
g = 2.002 					# spin g-factor for free electrons (for ferromag electrons)
h_bar = 1.054e-34			# reduced Planck's constant
m_e = 9.109e-31				# electron rest mass (kg)
N_Avo = 6.022e23			# Avogadro number
#################  TEMPERATURE and ARRAY SIZE: ##############################
T = 5
#N_2 = 200	      		# linear dimension of the lattice components, lattice-size= N_2 x N_2 (for both edge1 and edge2)
#N = 0


'''
						# N_2 = 50 for SPM case, 200 for ferro/ferri-mag case
'''
######### parameters for SUPERPARAMAGNETIC RUN: ######################
#J1 = 1.5
#J2 = 0
######### parameters for FERRO-/FERRIMAGNETIC RUN: ##########################

#J1_2 = 0.01
#J2_2 = 0.005			
'''
					# Good values are (for each case): 
									#J1_superparamag = 1.7
									#J2_superparamag = 0.0
									#J1_ferromag = 0.4
									#J2_ferromag = 0.003
									#J1_ferrimag = 0.4
									#J2_ferrimag = -0.001

									#N_superparamag = 50
									#N_ferromag = 200
									#N_ferrimag = 200
						# for sum of three types (for BVDGC fitting, SPM+ferro+ferri)
									#fa = 0.01
									#fb = 0.1
									#fc = 1-fa-fb		
									#divisor = 5
'''
######################################################################
J3_low = -24		#-31		# plus/minus3.13 Tesla, unit of J3 (magnetic field) used is Tesla,  B = 3.135 T on either side
J3_high = 24		#31
J3_int = 1
J3_divisor=10		# to make data more resolved, we divide the interval of the list/array J3 by this number, 10, 100, or so.
J3_multiplier = 1			# for superparamagnetic calculations, is 1 for ferrimagnetic/ferromagnetic calculationss
######################################################################
#f1 = 0.005			# fraction of signal from each measurement to be added to get final signal
#f2 = 1-f1
#divisor = 4			# factor to divide final graph values to obtain perfect fit with good y-axis match of experimental plot (because not all of the material in the real/synthesized carbon is graphitic/graphenic. Infact, this factor gives and idea about the fraction of material that is graphenic  (in the current case, 30May2021, BVDGC, divisor = 4)
######################################################################
_N_rows = 500   	# No. of max rows from top to be selected for plotting
num_skipped = 1  	# no. of rows skipped from top
################## range of axes: #######################
xlim_low =	-3.135
xlim_high = 3.135
ylim_low = -0.011
ylim_high = 0.011

