# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 16:46:44 2021

@author: OTPS
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import scipy as scipy
from CQED_fit import fit
from CQED_fit import avoided_crossing_direct_coupling_flat
from CQED_fit import avoided_crossing_direct_coupling
from CQED_fit import data_set
from CQED_fit import shortest_dist




def amp_to_volt(amp):
    

    #     amp in mV
    x = 0.15
    amp_fs = 2*2*amp/x/1e3 # in V!
    
    
    out_range = 0.750
    amp_HL = 5
    rect_wave = 1
    signal_voltage = rect_wave*amp_fs*out_range
    
    return signal_voltage
    
vpk_to_dbm = lambda v: 10*np.log10((v/np.sqrt(2))**2/(50*1e-3))

### DATA 211026 1014

I_211026_1014 = np.loadtxt('211026_1014_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R_211026_1014 = np.loadtxt('211026_1014_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod211026_1014 = np.sqrt(R_211026_1014**2 + I_211026_1014**2)

mod_mat211026_1014 = mod211026_1014.reshape(17,301)

nr_of_volt_measurements = np.arange(6.6, 7.4, 0.05).size

volt_measurements = np.arange(6.6, 7.4, 0.05)


# ### DATA 211026 1016 --> irregular measurement, needs to be discarded

# I_211026_1016 = np.loadtxt('211026_1016_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

# R_211026_1016 = np.loadtxt('211026_1016_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

# mod211026_1016 = np.sqrt(R_211026_1016**2 + I_211026_1016**2)

# mod_mat211026_1016 = mod211026_1016.reshape(17,301)

# nr_of_volt_measurements = np.arange(6.6, 7.4, 0.05).size

# volt_measurements = -np.arange(-6.6, -7.4, -0.05)


### DATA 211027 1005

I_211027_1005 = np.loadtxt('211027_1005_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R_211027_1005 = np.loadtxt('211027_1005_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod211027_1005 = np.sqrt(R_211027_1005**2 + I_211027_1005**2)

mod_mat211027_1005 = mod211027_1005.reshape(17,301)

nr_of_volt_measurements = np.arange(6.6, 7.4, 0.05).size

volt_measurements = -np.arange(-6.6, -7.4, -0.05)


### DATA 211027 1007

I_211027_1007 = np.loadtxt('211027_1007_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R_211027_1007 = np.loadtxt('211027_1007_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod211027_1007 = np.sqrt(R_211027_1007**2 + I_211027_1007**2)

mod_mat211027_1007 = mod211027_1007.reshape(17,301)

nr_of_volt_measurements = np.arange(6.6, 7.4, 0.05).size

volt_measurements = -np.arange(-6.6, -7.4, -0.05)



### DATA 211027 1008

I_211027_1008 = np.loadtxt('211027_1008_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R_211027_1008 = np.loadtxt('211027_1008_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod211027_1008 = np.sqrt(R_211027_1008**2 + I_211027_1008**2)

mod_mat211027_1008 = mod211027_1008.reshape(17,301)

nr_of_volt_measurements = np.arange(6.6, 7.4, 0.05).size

volt_measurements = -np.arange(-6.6, -7.4, -0.05)


### DATA 211029 1008

I_211029_1008 = np.loadtxt('211029_1008_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R_211029_1008 = np.loadtxt('211029_1008_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod211029_1008 = np.sqrt(R_211029_1008**2 + I_211029_1008**2)

mod_mat211029_1008 = mod211029_1008.reshape(17,301)


### DATA 211029 1009


I_211029_1009 = np.loadtxt('211029_1009_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R_211029_1009 = np.loadtxt('211029_1009_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod211029_1009 = np.sqrt(R_211029_1009**2 + I_211029_1009**2)

mod_mat211029_1009 = mod211029_1009.reshape(17,301)

### DATA 21112 1009


I_21112_1005 = np.loadtxt('21112_1005_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R_21112_1005 = np.loadtxt('21112_1005_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod21112_1005 = np.sqrt(R_21112_1005**2 + I_21112_1005**2)

mod_mat21112_1005 = mod21112_1005.reshape(17,301)



I_211025_1004 = np.loadtxt('211025_1004_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R_211025_1004 = np.loadtxt('211025_1004_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod211025_1004 = np.sqrt(R_211025_1004**2 + I_211025_1004**2)

mod_mat211025_1004 = mod211025_1004.reshape(21,101)



measurement_list = [mod_mat211029_1009, mod_mat211029_1008, mod_mat211027_1008, mod_mat211027_1007, mod_mat211026_1014, mod_mat21112_1005, mod_mat211027_1005]
index_list = [211029_1009, 211029_1008, 211027_1008, 211027_1007, 211026_1014, 21112_1005, 211027_1005]
amp_array = np.array([0.075, 0.125, 0.25, 0.5, 1, 1.5, 3])


volt_array = amp_to_volt(amp_array)
dbm_array = vpk_to_dbm(volt_array)

'''
For sqrt dependence of g
'''

array = volt_array**2 / 50 ### V^2 / ohm

# ### Find shortest distance

# cut_off = 45

# def shortest_dist(mod_mat, freq_start, freq_end, freq_steps, volt_start, volt_end, volt_steps, good_start, good_end):
    
#     nr_of_volt_measurements = np.arange(volt_start, volt_end, volt_steps).size
    
    
    
#     max_first_slice = max(mod_mat[0])
        
#     pos_max_first = np.where(mod_mat[0] == max_first_slice)[0][0]

#     freq_max_first = np.linspace(6.2e9, 6.8e9, 301)[pos_max_first]
    

    
    
        
#     # index_right = np.arange(pos_max_first, 301)
        
#     # index_left = np.arange(0, pos_max_first - cut_off)  # We need to exclude the leftover of the cavity resonance
    
    
    

    

    
    
#     distances = []
    
#     pairs = []
    
#     for i in range(good_start, nr_of_volt_measurements - good_end):
        
#         max_global = max(mod_mat[i])
       
#         pos_max_global = np.where(mod_mat[i] == max_global)[0][0]

#         freq_max_global = np.linspace(6.2e9, 6.8e9, 301)[pos_max_global]
        
#         # print('In iteration:', i, freq_max_global)
        
        
#         # new_slice_right = np.delete(mod_mat[i], index_left)
        
#         # new_slice_left = np.delete(mod_mat[i], index_right)
        
#         max_right = mod_mat[i][pos_max_first + 20:301].max()

#         pos_max_right = np.where(mod_mat[i] == max_right)[0][0]
        
#         freq_max_right = np.linspace(6.2e9, 6.8e9, 301)[pos_max_right]
        
#         # print('Right: ',freq_max_right)
        
        
        
#         max_left = mod_mat[i][0:(pos_max_first - cut_off)].max()
        
#         pos_max_left = np.where(mod_mat[i] == max_left)[0][0]
        
#         freq_max_left = np.linspace(6.2e9, 6.8e9, 301)[pos_max_left]
        
#         # print('Left: ',freq_max_left)

        
#         # if freq_max_global < freq_max_first:
            
#         #     maximum2 = max_right
            
#         # else:
            
#         #     maximum2 = max_left
        
        
        

#         pos_max1 = np.where(mod_mat[i] == max_left)[0][0]
#         pos_max2 = np.where(mod_mat[i] == max_right)[0][0]   

        
#         # pos_max1 = np.where(mod_mat[i] == max_global)[0][0]
#         # pos_max2 = np.where(mod_mat[i] == maximum2)[0][0]
        
#         # print('Max 1:',np.linspace(6.2e9, 6.8e9, 301)[pos_max1])
#         # print('Max 2:',np.linspace(6.2e9, 6.8e9, 301)[pos_max2])
        
#         pairs.append((pos_max1, pos_max2))
        
#         dist = np.abs(np.linspace(freq_start, freq_end, freq_steps)[pos_max2] - np.linspace(freq_start, freq_end, freq_steps)[pos_max1]) / 10**9
        

        
#         distances.append(dist)
        
    
#     # print(pairs)
#     # print(distances)
    
#     shortest_dist = min(distances)
    
#     pos_shortest_dist = distances.index(shortest_dist)
    
#     # print('Shortest dist. at volt iteration: ',volt_measurements[pos_shortest_dist + good_start])
    
#     # print('The shortest distance in the avoided crossing is (in [GHz]):', distances[pos_shortest_dist])
#     # print('Approximate coupling strength g (in [GHz]):', 0.5 * distances[pos_shortest_dist] )
    
#     points = [pos_max1, pos_max2]
    
#     # plt.plot(np.linspace(freq_start, freq_end, freq_steps), mod_mat[pos_shortest_dist + good_start], '-o', color='lightgrey', markevery=points)
#     # plt.title('Voltage slice of shortest distance')
#     # plt.plot()
    
#     g = 0.5 * distances[pos_shortest_dist]
    
#     return pairs, g


### Plot amp/g dependence

g_all = np.zeros(len(measurement_list))

print("Graphical g")

for i, data in enumerate(measurement_list):
    
    power = dbm_array[i]
    
    index = index_list[i]
    pairs_index, g_index = shortest_dist(data, 6.2e9, 6.8e9, 301, 6.6, 7.4, 0.05, 4, 3)
    
    print("g with input power", np.round(power,3),"is", np.round(g_index, 5),"in [GHz]")

    # print("g for experiment", index, "with input power", power,"is", g_index, "in [GHz]")
    
    g_all[i] = g_index
    


### Fit g data

def func(x, a, b, c):
    
    return a * x**2 + b * x + c


def sqrt(x, a, b, c):
    
    return x**(1/a) * b + c

def fit_g_data(func, x, y, a, b, c, method):
    
    popt, pcov = scipy.optimize.curve_fit(func, x, y, p0=[a, b, c], maxfev = 5000)

    string = ["graphical", "model"]
    string2 = ["Measured", "Fitted"]
    
    a, b, c = popt[0], popt[1], popt[2]
    
    a_err = np.sqrt(np.diag(pcov))[0]
    
    # print("a: ", a, " and b: ", b)
    

    plt.figure()

    plt.plot(x, y, '-o', color='lightgrey', label = f"{string2[method]} coupling strength g")
    
    plt.plot(x, func(x, popt[0], popt[1], popt[2]), 'red', label='Fit function f(x) = $x^{(1/a)} * b + c$')
    plt.xlabel('Linear signal input [$V^2/\Omega$]')
    plt.ylabel('g [GHz]')
    plt.legend(loc='best')
    plt.title(f"Coupling strength g ({string[method]} analysis)")
    plt.show()
    
    return a, a_err
    
def polyfit(x, y, d, method):
    
    params = np.polyfit(x, y, d)
    
    string = ["graphical", "model"]
    string2 = ["Measured", "Fitted"]
    
    
    # print("a: ", a, " and b: ", b)
    

    plt.figure()

    plt.plot(x, y, '-o', color='lightgrey', label = f"{string2[method]} coupling strength g")
    
    plt.plot(x, np.polyval(params, x), 'red', label=f"Polynomial fit of degree {d}")
    plt.xlabel('Signal input [Dbm]')
    plt.ylabel('g [GHz]')
    plt.legend(loc='best')
    plt.title(f"Coupling strength g ({string[method]} analysis)")
    plt.show()
    


### Pairs

def plot_upper_lower(data):

    pairs, g = shortest_dist(data, 6.2e9, 6.8e9, 301, 6.6, 7.4, 0.05, 4, 3)
    

    
    plt.figure()
    for n, tupel in enumerate(pairs):
        
         
        plt.plot(np.linspace(6.2e9, 6.8e9, 301)[tupel[0]], volt_measurements[n+4], '-o', color="lightgrey")
        plt.plot(np.linspace(6.2e9, 6.8e9, 301)[tupel[1]], volt_measurements[n+4], '-o', color="lightgrey")
    
    plt.title("Extracted intensity peaks (input power -66.478 dB)")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Abs. value voltage [V]")
    plt.show()

# plot_upper_lower(mod_mat211029_1009)


######################################################## Color map

'''
Digitize coil sweep
'''

def colorMap(data, dbm):

    Z = np.random.rand(17,301)
    
    
    fig, ax0 = plt.subplots()
    
    c = ax0.pcolor(data)
    ax0.set_title(f"Coil sweep for input power {dbm} dB")
    ax0.set_yticks(np.arange(1,18))
    ax0.set_yticklabels(np.round(np.linspace(6.6,7.4, 17),2))
    ax0.set_xticks(np.linspace(0, 301, 10))
    ax0.set_xticklabels(np.round(np.linspace(6.2e9, 6.8e9, 10)/10**9, 2))
    
    
    # c = ax1.pcolor(mod_mat, edgecolors='k', linewidths=4)
    # ax1.set_title('thick edges')
    
    
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Abs. value voltage [V]")
    fig.tight_layout()
    plt.show()
    
    
# colorMap(mod_mat211026_1014 / 10**9, 30)
# colorMap(mod_mat211026_1014, 30)

    
    
def colorMap_big_sweep(data, dbm):

    trans = data.transpose()
    
    fig, ax0 = plt.subplots()
    
    c = ax0.pcolor(data)
    ax0.set_title(f"Coil sweep for input power {dbm} dB")
    ax0.set_yticks(np.arange(1,22))
    ax0.set_yticklabels(np.round(np.linspace(-10,10, 21),2))
    ax0.set_xticks(np.linspace(0, 101, 10))
    ax0.set_xticklabels(np.round(np.linspace(6.2e9, 6.8e9, 10)/10**9, 2))
    
    # c = ax1.pcolor(mod_mat, edgecolors='k', linewidths=4)
    # ax1.set_title('thick edges')
    
    
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Voltage [V]")
    fig.tight_layout()
    plt.show()
    
colorMap_big_sweep(mod_mat211025_1004, -23.979)
    
    
'''
New color map to plot pairs on top
'''

def plot_upper_lower_color(data):

    pairs, g = shortest_dist(data, 6.2e9, 6.8e9, 301, 6.6, 7.4, 0.05, 4, 3)
    
    mod_mat = np.zeros(10*301).reshape(10,301)
    
    for n, tupel in enumerate(pairs):
        
        mod_mat[n][tupel[0]] = 1
        mod_mat[n][tupel[1]] = 1
        
    
    return mod_mat

    
color_peaks211029_1009 = plot_upper_lower_color(mod_mat211029_1009)

    


def colorMap_cut(data, pairs, dbm):

    
    fig, ax0 = plt.subplots()
    
    c = ax0.pcolor(data)
    # d = ax1.pcolor(pairs)
    # d = ax0.plot(np.linspace(6.2e9, 6.8e9, 301)/10**9, data[0])
    ax0.set_title(f"Coil sweep for input power {dbm} dB")
    ax0.set_yticks(np.arange(0,10))
    ax0.set_yticklabels(np.round(np.linspace(6.8,7.3, 10),2))
    ax0.set_xticks(np.linspace(0, 301, 10))
    ax0.set_xticklabels(np.round(np.linspace(6.2e9, 6.8e9, 10)/10**9, 2))
    
    # ax1.set_title(f"Extracted intensity peaks for input power {dbm} dB")
    # ax1.set_yticks(np.arange(0,10))
    # ax1.set_yticklabels(np.round(np.linspace(6.8,7.3, 10),2))
    # ax1.set_xticks(np.linspace(0, 301, 10))
    # ax1.set_xticklabels(np.round(np.linspace(6.2e9, 6.8e9, 10)/10**9, 2))
    
    # c = ax1.pcolor(mod_mat, edgecolors='k', linewidths=4)
    # ax1.set_title('thick edges')
    
    plt.xlabel("Frequency [GHz]")
    ax0.set_ylabel("Abs. value voltage [V]")
    # ax1.set_ylabel("Abs. value voltage [V]")
    # fig.set_size_inches(10, 9, forward=True)
    fig.tight_layout()
    
    

    plt.show()

# colorMap_cut(mod_mat211026_1014[4:14], color_peaks211029_1009, np.round(dbm_array[4], 3))



for data, dbm in zip(measurement_list, dbm_array):

    colorMap(data, np.round(dbm, 3))


##############################################################################

### Compare fits

g_fit = np.zeros(len(measurement_list))

print("Model g")

for i, data in enumerate(measurement_list):
    
    index = index_list[i]
    g_index, std_index = np.abs(fit(data)) / 10**9
    
    power = dbm_array[i]

    print("g with input power", np.round(power,3),"is", np.round(g_index, 5), "+-", np.round(std_index,4),"in [GHz]")
    
    g_fit[i] = g_index
    

a, a_err = fit_g_data(sqrt, array, g_all, 2, 1, 0, 0)
b, b_err = fit_g_data(sqrt, array, g_fit, 2, 1, 0, 1)

print(f"Graphical inverse exponent {a} with std. {a_err}")
print(f"Model inverse exponent {b} with std. {b_err}")



### Fine grid


# polyfit(dbm_array, g_fit, 3, 1)
# polyfit(dbm_array, g_all, 3, 0)


# #### Check with 8

# x = 7 + 4

# max_first_slice = max(mod_mat[0])
        
# pos_max_first = np.where(mod_mat[0] == max_first_slice)[0][0]

# freq_max_first = np.linspace(6.2e9, 6.8e9, 301)[pos_max_first]

# print('First: ',freq_max_first)

        
# index_right = np.arange(pos_max_first + 40, 301)
        
# index_left = np.arange(0, pos_max_first - 37)  # We need to exclude the leftover of the cavity resonance
    
# new_slice_right = np.delete(mod_mat[x], index_left)

# freq_right = np.linspace(6.2e9, 6.8e9, 301)[index_right]
        
# new_slice_left = np.delete(mod_mat[x], index_right)

# freq_left = np.linspace(6.2e9, 6.8e9, 301)[index_left]



# max_right = mod_mat[x][pos_max_first + 40:301].max()

# pos_max_right = np.where(mod_mat[x] == max_right)[0][0]

# freq_max_right = np.linspace(6.2e9, 6.8e9, 301)[pos_max_right]

# print('Right: ',freq_max_right)



# max_left = mod_mat[x][0:pos_max_first-40].max()

# pos_max_left = np.where(new_slice_left == max_left)[0][0]

# freq_max_left = np.linspace(6.2e9, 6.8e9, 301)[pos_max_left]

# print('Left: ',freq_max_left)


        

# max_global = max(mod_mat[x])
       
# pos_max_global = np.where(mod_mat[x] == max_global)[0][0]

# freq_max_global = np.linspace(6.2e9, 6.8e9, 301)[pos_max_global]

# print('Global: ',freq_max_global)

# distance = np.abs(freq_max_left - freq_max_global)

# g = 0.5 * distance / 10**9


