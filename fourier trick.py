import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.optimize import bisect
from scipy.signal import fftconvolve

mean1, sd1 = 3, 1.5
mean2, sd2 = -3, 1.5

# Defining two density functions
x1 = np.arange(mean1-4*sd1, mean1+4*sd1, step = 0.061)
y1 = norm.pdf(x1, mean1, sd1)

x2 = np.arange(mean2-4*sd2, mean2+4*sd2, step = 0.0223) # verander formaat en interval
y2 = norm.pdf(x2, mean2, sd2)



# Defining a common grid on which the densities are interpolated

boundary = 2*max(np.abs(min(min(x1), min(x2))),  max(x1) + max(x2))
common_grid = np.arange(-boundary, boundary, step=0.01)

# Make and apply interpolate functions
interpolate_f1 = interp1d(x1, y1, kind='linear', fill_value=0, bounds_error=False)
interpolate_f2 = interp1d(x2, y2, kind='linear', fill_value=0, bounds_error=False)
y1_interpolated = interpolate_f1(common_grid)
y2_interpolated = interpolate_f2(common_grid)

y12 = fftconvolve(y1_interpolated ,y2_interpolated , mode='same')
y12 /= np.trapz(y12, common_grid)

# Compute desired result
y12_true = norm.pdf(common_grid, mean1+mean2, np.sqrt(sd1**2+sd2**2))

plt.plot(common_grid,y1_interpolated, color='r', label='y1')
plt.plot(common_grid,y2_interpolated, color='r', label='y2')
plt.plot(common_grid, y12_true, color='b', label='y12_true')
plt.plot(common_grid, y12, label='y12_FFT')

plt.legend()
plt.show()

print(common_grid[np.argmax(y12)])
print(np.mean(np.abs(y12 - y12_true)))


