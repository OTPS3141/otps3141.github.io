# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 22:57:55 2021

@author: PC
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from uncertainties import ufloat
import pandas as pd 

x_axis_2 = np.linspace(6.4e9, 6.53e9, 66)

I_5 = np.loadtxt('1001_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R_5 = np.loadtxt('1001_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod_5 = np.sqrt(R_5**2 + I_5**2)

I = np.loadtxt('1008_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R = np.loadtxt('1008_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod = np.sqrt(R**2 + I**2)


x_axis = np.linspace(6.4e9, 6.6e9, 101)


# # Read data from file 'filename.csv' 
# # (in the same directory that your python process is based)
# # Control delimiters, rows, column names with read_csv (see later) 
# data1 = pd.read_csv("1001_IUHFLICh0cav_spec.dat", names = ['reeller Teil']) 
# data2 = pd.read_csv("1001_QUHFLICh0cav_spec.dat", names = ['imaginarer Teil'])

# #data = np.sqrt(data_real**2 + data_imaginar**2)
# real = np.array(data1[0:]).astype(np.float)
# imaginar = np.array(data2[0:]).astype(np.float)
# data = np.sqrt(real**2 + imaginar**2)
# #print (data)
# #"Frequency 1", "Start" -> 6.200000*^+9, "Stop" -> 6.800000*^+9, "Step size" -> 2.000000*^+6 |>, <| "Name" -> "DC Voltage 1", "Start" -> -6.600000*^+0, "Stop" -> -7.400000*^+0, "Step size" -> -5.000000*^-2 |>}, "DataInfo" -> <| "Measured Spots" -> 5.117000*^+3

# Number_diff_voltage = (7.4 - 6.6)/(5e-2) + 1
# #print(Number_diff_voltage)
# Number_plots_pe_voltage = np.size(real)/Number_diff_voltage 
# #print(Number_plots_pe_voltage)

# # Zu -6.6V
# data_voltage_1 = np.ravel( data[:301] )
# #print(data_voltage_1)

# #rechnen2 = (6.8-6.2)*1e9/(2e6) +1
# #print (rechnen2)
# frequency_range = np.linspace (6.2e9, 6.8e9, 301)
# print(frequency_range)

#def func_1(x, a,c,s):
    #return a*(1/(s*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-c)/s)**2)))

def func(x, a, b, c, d, e, f):
    return a/(b*(x-c)**2 + d*x + e) + f
#amp1 = 122.80
#cen1 = 49.90
#sigma1 = 11.78
[a, b, c, d, e, f], resl = curve_fit(func, x_axis, mod, p0=[1.2e7, 5.3e-5, 6.5e9, 3, 4, 5.4e-5])

print(a, b, c, d, e, f)
perrExp = np.sqrt(np.diag(resl))
print (perrExp)

y4 = func(x_axis, a, b, c, d, e, f)
# #y4 = func(frequency_range, 1e7, 5e-5, 6.5e9, 3, 4, 5e-5)
# #print(y4)
# def func_1(x, a1, b1, c1, d1, e1, f1, a2, b2, c2, d2, e2, f2):
#     return a1/(b1*(x-c1)**2 + d1*x + e1) + f1 + a2/(b2*(x-c2)**2 + d2*x + e2) + f2

# [a1, b1, c1, d1, e1, f1, a2, b2, c2, d2, e2, f2], resl = curve_fit(func_1, frequency_range, data_voltage_1, p0=[15469464367201.434, 588.5564899207855, 6495926402.396758, 5508659.51573197, 164.608402086788, 4.077948715351076e-05, 1e7, 5e-5, 6.5e9, 3, 4, 5e-5])
# #perrExp = np.sqrt(np.diag(resl))
# #print (perrExp)
# print(a1, b1, c1, d1, e1, f1, a2, b2, c2, d2, e2, f2)
# y5 = func_1(frequency_range, a1, b1, c1, d1, e1, f1, a2, b2, c2, d2, e2, f2)

plt.figure()

plt.plot(x_axis, mod, 'o', label = "original datapoints")
plt.plot(x_axis, y4, 'r-', label = "Fitted Cirve")
plt.xlabel('frequency[Hz]')
plt.ylabel('Amplitude[m]')

# plt.figure()
# plt.plot(frequency_range, data_voltage_1, 'o', label = "original datapoints")
# plt.plot(frequency_range, y5, 'r-', label = "Fitted Cirve")
# #plt.yticks( np.linspace(0.00005,0.0001, 100 ) )
# plt.xlabel('frequency[Hz]')
# plt.ylabel('Amplitude[m]')
# #plt.plot(frequency_range, y5, 'r-', label = "Fitted Cirve")
