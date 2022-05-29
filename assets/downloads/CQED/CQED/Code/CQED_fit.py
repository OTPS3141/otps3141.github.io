# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 15:37:09 2021

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
# from colormap import shortest_dist
# from colormap import colorMap

### DATA 211026 1014

I = np.loadtxt('211026_1014_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R = np.loadtxt('211026_1014_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod = np.sqrt(R**2 + I**2)

mod_mat = mod.reshape(17,301)

nr_of_volt_measurements = np.arange(6.6, 7.4, 0.05).size

volt_measurements = np.arange(6.6, 7.4, 0.05)

good_start = 4
good_end = 3


### Find shortest distance

cut_off = 45

def shortest_dist(mod_mat, freq_start, freq_end, freq_steps, volt_start, 
                  volt_end, volt_steps, good_start, good_end):
    
    nr_of_volt_measurements = np.arange(volt_start, volt_end, volt_steps).size
    
    
    
    max_first_slice = max(mod_mat[0])
        
    pos_max_first = np.where(mod_mat[0] == max_first_slice)[0][0]

    freq_max_first = np.linspace(6.2e9, 6.8e9, 301)[pos_max_first]
    

    
    
        
    # index_right = np.arange(pos_max_first, 301)
        
    # index_left = np.arange(0, pos_max_first - cut_off)  # We need to exclude the leftover of the cavity resonance
    
    
    

    

    
    
    distances = []
    
    pairs = []
    
    for i in range(good_start, nr_of_volt_measurements - good_end):
        
        max_global = max(mod_mat[i])
       
        pos_max_global = np.where(mod_mat[i] == max_global)[0][0]

        freq_max_global = np.linspace(6.2e9, 6.8e9, 301)[pos_max_global]
        
        # print('In iteration:', i, freq_max_global)
        
        
        # new_slice_right = np.delete(mod_mat[i], index_left)
        
        # new_slice_left = np.delete(mod_mat[i], index_right)
        
        max_right = mod_mat[i][pos_max_first + 20:301].max()

        pos_max_right = np.where(mod_mat[i] == max_right)[0][0]
        
        freq_max_right = np.linspace(6.2e9, 6.8e9, 301)[pos_max_right]
        
        # print('Right: ',freq_max_right)
        
        
        
        max_left = mod_mat[i][0:(pos_max_first - cut_off)].max()
        
        pos_max_left = np.where(mod_mat[i] == max_left)[0][0]
        
        freq_max_left = np.linspace(6.2e9, 6.8e9, 301)[pos_max_left]
        
        # print('Left: ',freq_max_left)

        
        # if freq_max_global < freq_max_first:
            
        #     maximum2 = max_right
            
        # else:
            
        #     maximum2 = max_left
        
        
        

        pos_max1 = np.where(mod_mat[i] == max_left)[0][0]
        pos_max2 = np.where(mod_mat[i] == max_right)[0][0]   

        
        # pos_max1 = np.where(mod_mat[i] == max_global)[0][0]
        # pos_max2 = np.where(mod_mat[i] == maximum2)[0][0]
        
        # print('Max 1:',np.linspace(6.2e9, 6.8e9, 301)[pos_max1])
        # print('Max 2:',np.linspace(6.2e9, 6.8e9, 301)[pos_max2])
        
        pairs.append((pos_max1, pos_max2))
        
        dist = np.abs(np.linspace(freq_start, freq_end, freq_steps)[pos_max2] - 
            np.linspace(freq_start, freq_end, freq_steps)[pos_max1]) / 10**9
        

        
        distances.append(dist)
        
    
    # print(pairs)
    # print(distances)
    
    shortest_dist = min(distances)
    
    pos_shortest_dist = distances.index(shortest_dist)
    
    # print('Shortest dist. at volt iteration: ',volt_measurements[pos_shortest_dist + good_start])
    
    # print('The shortest distance in the avoided crossing is (in [GHz]):', distances[pos_shortest_dist])
    # print('Approximate coupling strength g (in [GHz]):', 0.5 * distances[pos_shortest_dist] )
    
    points = [pos_max1, pos_max2]
    
    # plt.plot(np.linspace(freq_start, freq_end, freq_steps), mod_mat[pos_shortest_dist + good_start], '-o', color='lightgrey', markevery=points)
    # plt.title('Voltage slice of shortest distance')
    # plt.plot()
    
    g = 0.5 * distances[pos_shortest_dist]
    
    return pairs, g


def data_set(data):

    pairs, g_approx = shortest_dist(data, 6.2e9, 6.8e9, 301, 6.6, 7.4, 0.05, good_start, good_end)
    
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
    
    
    
    for i, tupel in enumerate(new_pairs):            
        
    
        
        upper_freqs[i] = np.linspace(6.2e9, 6.8e9, 301)[tupel[1]]
        lower_freqs[i] = np.linspace(6.2e9, 6.8e9, 301)[tupel[0]]
    

    return lower_freqs, upper_freqs, g_approx

    
### Fitting

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

'''
Model end
'''


def fit(data):
    
    ### Data initialising
    
    lower_freqs = data_set(data)[0]
    upper_freqs = data_set(data)[1]
    
    g_approx = data_set(data)[2]
    
    x = volt_measurements[good_start:-good_end]

    f_data = np.zeros([len(x), 2])
    f_data[:,0] = lower_freqs
    f_data[:,1] = upper_freqs
    

    upper_flux = x[6:]
    lower_flux = x[:6]  
    
    g_guess = g_approx * (10**9)
    
    f1_guess = np.mean(lower_freqs) - 2.2e9
    f2_guess = np.mean(upper_freqs) - 2.2e9
    
    c1_guess=(upper_freqs[-1]-upper_freqs[0])/\
    (upper_flux[-1]-upper_flux[0])
    c2_guess=(lower_freqs[-1]-lower_freqs[0])/\
    (lower_flux[-1]-lower_flux[0])
    
    freqs = avoided_crossing_direct_coupling(x, f1_guess, f2_guess, c1_guess, c2_guess, 0.126e9, flux_state=0)


    
    ### Flatten
    
    freqs_flat = freqs.flatten()
    
    popt, pcov = scipy.optimize.curve_fit(avoided_crossing_direct_coupling_flat, x, f_data.flatten(), p0=[f1_guess, f2_guess, c1_guess, c2_guess, 0.126e9], maxfev=5000)
    perr = np.sqrt(np.diag(pcov))
    # print(popt)
    
    g = popt[4]
    std = perr[4]
    # plt.figure()
        
    # plt.plot(x, lower_freqs, '-o', color='lightgrey', label = 'Observed frequency sweep')
    # plt.plot(x, upper_freqs, '-o', color='lightgrey')
        
        
    
    # plt.plot(x, avoided_crossing_direct_coupling(x, popt[0], popt[1], popt[2], popt[3],
    #                                     popt[4])[:,0], 'red', label='Fit upper frequencies')
    # plt.plot(x, avoided_crossing_direct_coupling(x, popt[0], popt[1], popt[2], popt[3],
    #                                     popt[4])[:,1], 'blue', label='Fit lower frequencies')    
    # plt.legend(loc='best')

    # print("The absolute value of the estimated g is: ", np.abs(popt[4]))
    # plt.title("Avoided crossing fit")
    # plt.xlabel("Modulus coil voltage [V]")
    # plt.ylabel("Resonance frequency [GHz]")
    # plt.show()
    
    return g, std
    
                                                                                        

print(fit(mod_mat))