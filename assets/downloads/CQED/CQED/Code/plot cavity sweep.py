# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 07:26:19 2021

@author: OTPS
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
from scipy import optimize
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import matplotlib.ticker as ticker
from lorentzian import fit_lorentz_3 
from lorentzian import _3Lorentzian


### Data


I = np.loadtxt('1008_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R = np.loadtxt('1008_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod = np.sqrt(R**2 + I**2)


x_axis = np.linspace(6.4e9, 6.6e9, 101)



I_2 = np.loadtxt('1000_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R_2 = np.loadtxt('1000_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod_2 = np.sqrt(R_2**2 + I_2**2)


x_axis_2 = np.linspace(6.4e9, 6.53e9, 66)



I_3 = np.loadtxt('1005_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R_3 = np.loadtxt('1005_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod_3 = np.sqrt(R_3**2 + I_3**2)





I_4 = np.loadtxt('1004_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R_4 = np.loadtxt('1004_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod_4 = np.sqrt(R_4**2 + I_4**2)



I_5 = np.loadtxt('1001_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R_5 = np.loadtxt('1001_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod_5 = np.sqrt(R_5**2 + I_5**2)

plt.figure()
plt.plot(x_axis, mod, "-o", color="lightgrey", label=r"Frequency sweep for ""\n""input power 1.6 mW")
plt.legend(loc="best")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude [V]")
plt.show()

input_power = [0.016, 0.036, 0.064, 0.1]
data = [mod_2, mod_3, mod_4, mod_5]

for i, power in enumerate(input_power):
    
    i = i + 2
    

    plt.figure()
    
    plt.plot(x_axis_2, data[i-2], "-o", color="lightgrey", label=rf"Frequency sweep for input power {power} mW")
    plt.legend(loc="lower center")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [V]")
    plt.show()
    
    print(i, power)

