import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
from scipy import optimize
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import matplotlib.ticker as ticker

### Multilorentzian function:
    
def _3Lorentzian(x, amp1, cen1, wid1, amp2,cen2,wid2, amp3, cen3, wid3):
    return (amp1*wid1**2/((x-cen1)**2+wid1**2)) +\
            (amp2*wid2**2/((x-cen2)**2+wid2**2)) +\
                (amp3*wid3**2/((x-cen3)**2+wid3**2))
    
def _2Lorentzian(x, amp1, cen1, wid1, amp2,cen2,wid2):
    return (amp1*wid1**2/((x-cen1)**2+wid1**2)) +\
            (amp2*wid2**2/((x-cen2)**2+wid2**2))

def Lorentzian(x, amp, cen, wid, A):
    return amp*wid**2/((x-cen)**2+wid**2) + A




### Data

I = np.loadtxt('1008_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R = np.loadtxt('1008_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod = np.sqrt(R**2 + I**2)

print(mod.shape)

x_axis = np.linspace(6.4e9, 6.6e9, 101)



I_2 = np.loadtxt('1001_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R_2 = np.loadtxt('1001_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod_2 = np.sqrt(R_2**2 + I_2**2)

print(mod_2.shape)

x_axis_2 = np.linspace(6.4e9, 6.53e9, 66)




I_3 = np.loadtxt('211026_1014_QUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

R_3 = np.loadtxt('211026_1014_IUHFLICh0cav_spec.dat', unpack = True, skiprows=1)

mod_3 = np.sqrt(R_3**2 + I_3**2)

x_axis_3 = np.linspace(6.4e9, 6.6e9, 301)


mod_mat = mod_3.reshape(17,301)

nr_of_volt_measurements = np.arange(6.6, 7.4, 0.05).size

volt_measurements = np.arange(6.6, 7.4, 0.05)

plt.figure()
plt.plot(x_axis, mod)
plt.show()




# ### Fit 1 Lorentzian


def fit_lorentz_1(func, x, y, amp, cen, wid, A):
    
    popt_Lorentzian, pcov_Lorentzian = scipy.optimize.curve_fit(func, x, y, p0=[amp, cen, wid, A])
    
    print("Parameters simple: ",popt_Lorentzian)
    
    perr_1lorentz = np.sqrt(np.diag(pcov_Lorentzian))
    print("Errors simple: ", perr_1lorentz)
    
    plt.figure()
    
    plt.plot(x, y, '-o', color='lightgrey', label = 'Observed frequency sweep')
    plt.title('Simple Lorentzian fit to frequency sweep')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [V]')
    plt.grid(True)
    plt.plot(x, Lorentzian(x_axis, popt_Lorentzian[0], popt_Lorentzian[1], popt_Lorentzian[2], popt_Lorentzian[3]), 'red', label='Simple Lorentzian fit')
    plt.legend(loc='best')

    plt.show()

# ### Fit 2 Lorentzian

def fit_lorentz_2(func, x, y, amp1, cen1, wid1, amp2, cen2, wid2):

    popt_2lorentz, pcov_2lorentz = scipy.optimize.curve_fit(_2Lorentzian, x_axis_2, mod_2, p0=[0.0006, 6.44e9, 0.01e9, \
                                                                                        0.0007, 6.465e9, 0.02e9])
    
    print(popt_2lorentz)
    
    plt.figure()
    
    plt.plot(x_axis_2, mod_2, '-o', color='lightgrey', label = 'Observed frequency sweep')
    
    
    plt.plot(x_axis_2, _2Lorentzian(x_axis_2, popt_2lorentz[0], popt_2lorentz[1], popt_2lorentz[2], popt_2lorentz[3],
                                    popt_2lorentz[4], popt_2lorentz[5]), 'red', label='Multi Lorentzian fit')
    plt.legend(loc='best')
    
    plt.show()
    
    




def fit_lorentz_3(func, x, y, amp1, cen1, wid1, amp2, cen2, wid2, amp3, cen3, wid3):


    popt_3lorentz, pcov_3lorentz = scipy.optimize.curve_fit(func, x, y, p0=[amp1, cen1, wid1, amp2, cen2, wid2, amp3, cen3, wid3])
    
    print('Parameters:',popt_3lorentz)
    
    perr_3lorentz = np.sqrt(np.diag(pcov_3lorentz))
    print('Uncertainties:', perr_3lorentz)
    
    plt.figure()
    
    plt.plot(x, y, '-o', color='lightgrey', label = 'Observed frequency sweep')
    
    
    plt.plot(x, func(x, popt_3lorentz[0], popt_3lorentz[1], popt_3lorentz[2], popt_3lorentz[3],
                                    popt_3lorentz[4], popt_3lorentz[5], popt_3lorentz[6], popt_3lorentz[7], popt_3lorentz[8]), 'red', label='Multi Lorentzian fit')
    plt.legend(loc='best')
    plt.title('Multi Lorentzian fit to frequency sweep')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [V]')
    plt.show()


# fit_lorentz_3(_3Lorentzian, x_axis_3, mod_3[6], 0.00018, 6.29e9, 0.01e9, 0.0001, 6.44e9, 0.02e9, 0.0002, 6.55e9, 0.02e9)

fit_lorentz_3(_3Lorentzian, x_axis_2, mod_2, 0.00035, 6.42e9, 0.01e9, 0.0006, 6.44e9, 0.02e9, 0.0008, 6.46e9, 0.02e9)

fit_lorentz_1(Lorentzian, x_axis, mod, 0.04, 6.49e9, 0.01e9, 0.0001)



# fit_lorentz_2(_2Lorentzian, x_axis_3, mod_3, 0.00065, 6.443e9, 0.01e9, 0.00075, 6.46e9, 0.01e9)
