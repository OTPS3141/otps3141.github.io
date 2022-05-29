# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:35:57 2021

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
from colormap import shortest_dist
from colormap import colorMap

"""
### Data
"""

### DATA 211026 1014


I = np.loadtxt('211029_1009_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R = np.loadtxt('211029_1009_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod = np.sqrt(R**2 + I**2)

mod_mat = mod.reshape(17,301)

nr_of_volt_measurements = np.arange(6.6, 7.4, 0.05).size

volt_measurements = np.arange(6.6, 7.4, 0.05)


# ### DATA 211026 1016


# I = np.loadtxt('211026_1016_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

# R = np.loadtxt('211026_1016_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

# mod = np.sqrt(R**2 + I**2)

# mod_mat = mod.reshape(17,301)

# nr_of_volt_measurements = np.arange(6.6, 7.4, 0.05).size

# volt_measurements = np.arange(6.6, 7.4, 0.05)

# colorMap()


"""
### Run function

"""

good_start = 4
good_end = 3

pairs, g_approx = shortest_dist(mod_mat, 6.2e9, 6.8e9, 301, 6.6, 7.4, 0.05, 4, 3)

### Order pairs --> smallest freq left

def order_tuple(tuple):
    
    ordered_tuple_list = []
    
    for i, tupel in enumerate(pairs):
        
        if tupel[0] > tupel[1]:
            
            ordered_tuple_list.append((tupel[1], tupel[0]))
        
        else: 
            
            ordered_tuple_list.append((tupel[0], tupel[1]))
            
    return ordered_tuple_list

new_pairs = order_tuple(pairs)

### Get upper/lower freq

upper_freqs = np.zeros(volt_measurements.size-(good_start + good_end))
lower_freqs = np.zeros(volt_measurements.size-(good_start + good_end))

upper_freqs_index = []
lower_freqs_index = []


for i, tupel in enumerate(new_pairs):            
    
    # upper_freqs_index[i] = tupel[1]
    # lower_freqs_index[i] = tupel[0]
    
    upper_freqs[i] = np.linspace(6.2e9, 6.8e9, 301)[tupel[1]]
    lower_freqs[i] = np.linspace(6.2e9, 6.8e9, 301)[tupel[0]]




    



"""
### Model
"""

def avoided_crossing_direct_coupling(flux, f_center1, f_center2,
                                     c1, c2, g, flux_state=0):
    """
    Calculates the frequencies of an avoided crossing for the following model.
        [f_1, g ]
        [g,  f_2]

    f1 = c1*flux + f_center1
    f2 = c2*flux + f_center2
	
    flux: is the array of voltages
	Data that you want to fit is the array (len(voltages, 2)) of frequencies corresponding to these voltages
	
    g:  the coupling strength, beware to relabel your variable if using this
        model to fit J1 or J2.
    flux_state:  this is a switch used for fitting. It determines which
        transition to return
    """

    if type(flux_state) == int:
        flux_state = [flux_state] * len(flux)

    frequencies = np.zeros([len(flux), 2])
    
    for kk, dac in enumerate(flux):
        f_1 = dac * c1 + f_center1
        f_2 = dac * c2 + f_center2
        
        # f_1 = dac**(2) + f_center1
        # f_2 = dac**(0.5) + f_center2
        matrix = [[f_1, g],
                  [g, f_2]]
        frequencies[kk, :] = np.linalg.eigvalsh(matrix)[:2]
    # result = np.where(flux_state, frequencies[:, 0], frequencies[:, 1])
    return frequencies


##################





### Estimate c1, c2


# def linear_func(x, a, b, c):
    
#     return a * x**c + b

# def fitting_c1():
    
#     x = lower_freqs
    
#     y = flux
    
#     params, err = scipy.optimize.curve_fit(linear_func, x, y, p0=[1, f1_guess, 2])
    
#     return [params[0], params[1], params[2]]

# def fitting_c2():
    
#     x = upper_freqs
    
#     y = flux
    
#     params, err = scipy.optimize.curve_fit(linear_func, x, y, p0=[3, f1_guess, 0.5])
    
#     return [params[0], params[1], params[2]]



### Parameter

good_start = 4
good_end = 3

flux = volt_measurements[good_start:-good_end]

upper_flux = flux[6:]
lower_flux = flux[:6]  ### What is upper, lower_flux?

g_guess = g_approx * (10**9)

f1_guess = np.mean(lower_freqs) - 2.2e9
f2_guess = np.mean(upper_freqs) - 2.2e9

c1_guess=(upper_freqs[-1]-upper_freqs[0])/\
(upper_flux[-1]-upper_flux[0])
c2_guess=(lower_freqs[-1]-lower_freqs[0])/\
(lower_flux[-1]-lower_flux[0])

# c1_guess = -0.2e9/(7.4-6.8)
# c2_guess = -0.18e9/(7.4-6.8)

# c1_fit = fitting_c1()
# c2_fit = fitting_c2()

# c1_guess = c1_fit
# c2_guess = c2_fit


freqs = avoided_crossing_direct_coupling(flux, f1_guess, f2_guess, 0.2e9/(7.4-6.8), 0.18e9/(7.4-6.8), 0.126e9, flux_state=0)

# result_fit, freqs_fit = avoided_crossing_direct_coupling(flux, f1_guess, f2_guess, c1_fit, c2_fit, g_guess)


print('Estimated freqs: ', freqs)



freq_dist = freqs[:,1] - freqs[:,0]



print('Mean g [GHz]: ', 0.5*min(freq_dist)/10**9)


### Figure


# plt.figure()

# for n, tupel in enumerate(new_pairs):
    
     
#     plt.plot(np.linspace(6.2e9, 6.8e9, 301)[tupel[0]], volt_measurements[n+4], '-o', color='red')
#     plt.plot(np.linspace(6.2e9, 6.8e9, 301)[tupel[1]], volt_measurements[n+4], '-o', color='blue')
 
# plt.title('Data of lower and upper frequencies')    
# plt.ylabel('Absolute value of flux [V]')
# plt.xlabel('Frequencies [GHz]')


### Plot estimated freqs


# for n, tupel in enumerate(freqs):
    
    
#     plt.plot(tupel[0], volt_measurements[n+4], '-+', color='red')
#     plt.plot(tupel[1], volt_measurements[n+4], '-+', color='blue')



# plt.title('Fitting results of lower and upper frequencies')    
# plt.ylabel('Absolute value of flux [V]')
# plt.xlabel('Frequencies [GHz]')


### Fitting function

'''
x-data: volt_measurements[4:]

y-data: lower_freqs

'''


def avoided_crossing_direct_coupling_lower(flux, f_center1, f_center2,
                                     c1, c2, g, flux_state=0):
    """
    Calculates the frequencies of an avoided crossing for the following model.
        [f_1, g ]
        [g,  f_2]

    f1 = c1*flux + f_center1
    f2 = c2*flux + f_center2
	
    flux: is the array of voltages
	Data that you want to fit is the array (len(voltages, 2)) of frequencies corresponding to these voltages
	
    g:  the coupling strength, beware to relabel your variable if using this
        model to fit J1 or J2.
    flux_state:  this is a switch used for fitting. It determines which
        transition to return
    """

    if type(flux_state) == int:
        flux_state = [flux_state] * len(flux)

    frequencies = np.zeros([len(flux), 2])
    
    for kk, dac in enumerate(flux):
        f_1 = dac * c1 + f_center1
        f_2 = dac * c2 + f_center2
        
        # f_1 = dac**(2) + f_center1
        # f_2 = dac**(0.5) + f_center2
        matrix = [[f_1, g],
                  [g, f_2]]
        frequencies[kk, :] = np.linalg.eigvalsh(matrix)[:2]
    # result = np.where(flux_state, frequencies[:, 0], frequencies[:, 1])
    return frequencies[:,0]

def avoided_crossing_direct_coupling_upper(flux, f_center1, f_center2,
                                     c1, c2, g, flux_state=0):
    """
    Calculates the frequencies of an avoided crossing for the following model.
        [f_1, g ]
        [g,  f_2]

    f1 = c1*flux + f_center1
    f2 = c2*flux + f_center2
	
    flux: is the array of voltages
	Data that you want to fit is the array (len(voltages, 2)) of frequencies corresponding to these voltages
	
    g:  the coupling strength, beware to relabel your variable if using this
        model to fit J1 or J2.
    flux_state:  this is a switch used for fitting. It determines which
        transition to return
    """

    if type(flux_state) == int:
        flux_state = [flux_state] * len(flux)

    frequencies = np.zeros([len(flux), 2])
    
    for kk, dac in enumerate(flux):
        f_1 = dac * c1 + f_center1
        f_2 = dac * c2 + f_center2
        
        # f_1 = dac**(2) + f_center1
        # f_2 = dac**(0.5) + f_center2
        matrix = [[f_1, g],
                  [g, f_2]]
        frequencies[kk, :] = np.linalg.eigvalsh(matrix)[:2]
    # result = np.where(flux_state, frequencies[:, 0], frequencies[:, 1])
    return frequencies[:,1]

def avoided_crossing_direct_coupling_flat(flux, f_center1, f_center2,
                                     c1, c2, g, flux_state=0):
    """
    Calculates the frequencies of an avoided crossing for the following model.
        [f_1, g ]
        [g,  f_2]

    f1 = c1*flux + f_center1
    f2 = c2*flux + f_center2
	
    flux: is the array of voltages
	Data that you want to fit is the array (len(voltages, 2)) of frequencies corresponding to these voltages
	
    g:  the coupling strength, beware to relabel your variable if using this
        model to fit J1 or J2.
    flux_state:  this is a switch used for fitting. It determines which
        transition to return
    """

    if type(flux_state) == int:
        flux_state = [flux_state] * len(flux)

    frequencies = np.zeros([len(flux), 2])
    
    for kk, dac in enumerate(flux):
        f_1 = dac * c1 + f_center1
        f_2 = dac * c2 + f_center2
        
        # f_1 = dac**(2) + f_center1
        # f_2 = dac**(0.5) + f_center2
        matrix = [[f_1, g],
                  [g, f_2]]
        frequencies[kk, :] = np.linalg.eigvalsh(matrix)[:2]
    # result = np.where(flux_state, frequencies[:, 0], frequencies[:, 1])
    return frequencies.flatten()


x = flux

f_data = np.zeros([len(x), 2])
f_data[:,0] = lower_freqs
f_data[:,1] = upper_freqs

def fit():

    x = flux
    
    # print(f_data)
        
    # popt_lower, pcov_lower = scipy.optimize.curve_fit(avoided_crossing_direct_coupling_lower, x, lower_freqs, p0=[f1_guess, f2_guess, c1_guess, c2_guess, 0.126e9], maxfev=5000)
    # popt_upper, pcov_upper = scipy.optimize.curve_fit(avoided_crossing_direct_coupling_upper, x, upper_freqs, p0=[f1_guess, f2_guess, c1_guess, c2_guess, 0.126e9], maxfev=5000)

    # print(popt_lower)
    # print(popt_upper)
    
    ### Flatten
    
    freqs_flat = freqs.flatten()
    
    popt, pcov = scipy.optimize.curve_fit(avoided_crossing_direct_coupling_flat, x, f_data.flatten(), p0=[f1_guess, f2_guess, c1_guess, c2_guess, 0.126e9], maxfev=5000)

    
    
   
    
    
        
    # plt.plot(x, lower_freqs, '-o', color='lightgrey', label = 'Observed frequency sweep')
    # plt.plot(x, upper_freqs, '-o', color='lightgrey')
        
    plt.plot(lower_freqs, x, '-o', color='lightgrey', label = 'Observed frequency sweep')
    plt.plot(upper_freqs, x, '-o', color='lightgrey')
        

    plt.plot(avoided_crossing_direct_coupling(x, popt[0], popt[1], popt[2], popt[3],
                                        popt[4])[:,0], x, 'red', label='Fit lower frequencies $f_-$')
    plt.plot(avoided_crossing_direct_coupling(x, popt[0], popt[1], popt[2], popt[3],
                                        popt[4])[:,1], x, 'blue', label='Fit upper frequencies $f_+$')
    
    lower_fit = avoided_crossing_direct_coupling(x, popt[0], popt[1], popt[2], popt[3],
                                        popt[4])[:,0]
    
    upper_fit = avoided_crossing_direct_coupling(x, popt[0], popt[1], popt[2], popt[3],
                                        popt[4])[:,1]
    # plt.plot(x, avoided_crossing_direct_coupling(x, popt[0], popt[1], popt[2], popt[3],
    #                                     popt[4])[:,0], 'red', label='Fit upper frequencies')
    # plt.plot(x, avoided_crossing_direct_coupling(x, popt[0], popt[1], popt[2], popt[3],
    #                                     popt[4])[:,1], 'blue', label='Fit lower frequencies')    
    plt.legend(loc='upper center')

    print("The absolute value of the estimated g is: ", np.abs(popt[4]) / 10**6, "[MHz]")
    plt.title("Avoided crossing fit for input power -23.979 dB")
    plt.ylabel("Abs. value voltage [V]")
    plt.xlabel("Resonance frequency [GHz]")
    plt.grid(True)
    # plt.axis('off')
    plt.show()
    
    return popt


fit()

### Plot fit

popt = fit()

lower_fit = np.round(avoided_crossing_direct_coupling(x, popt[0], popt[1], popt[2], popt[3],
                                        popt[4])[:,0] / 10**9, 2)

upper_fit = np.round(avoided_crossing_direct_coupling(x, popt[0], popt[1], popt[2], popt[3],
                                        popt[4])[:,1] / 10**9, 2)



freq_axis = np.round(np.linspace(6.2, 6.8, 301), 3)

x_array_lower = np.zeros(10)
x_array_upper = np.zeros(10)


for i in range(0, 10):
    
    index_lower = np.where(freq_axis == lower_fit[i])
    index_upper = np.where(freq_axis == upper_fit[i])
    
    # print(index_lower[0][0])
    
    x_array_lower[i] = index_lower[0][0]
    x_array_upper[i] = index_upper[0][0]

    



def colorMap_fit(data, dbm):
    
    trans = data.transpose()

    
    
    fig, ax0 = plt.subplots()
    
    # c = ax0.pcolor(trans)
    c = ax0.pcolor(data)
    ax0.set_title(f"Coil sweep for input power {dbm} dB")
    ax0.set_yticks(np.arange(0,10))
    # # ax0.set_xticks(np.linspace(6.8, 7.25, 10))
    ax0.set_yticklabels(np.round(np.linspace(6.8,7.25, 10),2))
    ax0.set_xticks(np.linspace(1, 301, 10))
    ax0.set_xticklabels(np.round(np.linspace(6.2e9, 6.8e9, 10)/10**9, 2))
    
    # ax0.set_xticks(np.linspace(6.2e9, 6.8e9, 301))
    # ax0.set_yticks(np.linspace(6.8, 7.25, 10))
    
    # ax0.plot(np.arange(0, 10), x_array_lower, 'o-', color = 'red', label='Fit lower frequencies $f_-$')
    # ax0.plot(np.arange(0, 10), x_array_upper, 'o-', color = 'blue', label='Fit upper frequencies $f_+$')
    
    
    ax0.plot(x_array_lower, np.arange(0, 10), 'o-', color = 'red', label='Fit lower frequencies $f_-$', linewidth=1.0)
    ax0.plot(x_array_upper, np.arange(0, 10), 'o-', color = 'blue', label='Fit upper frequencies $f_+$', linewidth=1.0)
    

    
    
    
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Abs. value voltage [V]")
    plt.legend(loc='upper center')
    fig.tight_layout()
    plt.show() 

colorMap_fit(mod_mat[4:14], -23.979)    


### Plot peaks

x_freq_lower = np.zeros(10)
x_freq_upper = np.zeros(10)

lower_freqs = lower_freqs / 10**9
upper_freqs = upper_freqs / 10**9



for i in range(0, 10):
    
    index_lower = np.where(freq_axis == lower_freqs[i])
    index_upper = np.where(freq_axis == upper_freqs[i])
    
    # print(index_lower[0][0])
    
    x_freq_lower[i] = index_lower[0][0]
    x_freq_upper[i] = index_upper[0][0]
                                                                                   
def colorMap_peaks(data, dbm):
    
    trans = data.transpose()

    
    
    fig, ax0 = plt.subplots()
    
    # c = ax0.pcolor(trans)
    c = ax0.pcolor(data)
    ax0.set_title(f"Coil sweep for input power {dbm} dB")
    ax0.set_yticks(np.arange(0,10))
    # # ax0.set_xticks(np.linspace(6.8, 7.25, 10))
    ax0.set_yticklabels(np.round(np.linspace(6.8,7.25, 10),2))
    ax0.set_xticks(np.linspace(1, 301, 10))
    ax0.set_xticklabels(np.round(np.linspace(6.2e9, 6.8e9, 10)/10**9, 2))
    
    # ax0.set_xticks(np.linspace(6.2e9, 6.8e9, 301))
    # ax0.set_yticks(np.linspace(6.8, 7.25, 10))
    
    # ax0.plot(np.arange(0, 10), x_array_lower, 'o-', color = 'red', label='Fit lower frequencies $f_-$')
    # ax0.plot(np.arange(0, 10), x_array_upper, 'o-', color = 'blue', label='Fit upper frequencies $f_+$')
    
    
    ax0.plot(x_freq_lower, np.arange(0, 10), 'o', color = 'red', label='Extracted lower frequencies')
    ax0.plot(x_freq_upper, np.arange(0, 10), 'o', color = 'blue', label='Extracted upper frequencies')
    

    
    
    
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Abs. value voltage [V]")
    plt.legend(loc='upper center')
    fig.tight_layout()
    plt.show() 

colorMap_peaks(mod_mat[4:14], -23.979)    
