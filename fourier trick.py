import numpy as np
import pandas as pd
import seaborn as sns


from scipy.fft import fft, ifft
from scipy.signal import fftconvolve

x1 = np.linspace(-10,10,50000)
x2 = np.linspace(-4,4,10000)

dx1 = x1[1]-x1[0]
dx2 = x2[1]-x2[0]

y1 = norm.pdf(x=x1, loc=-5, scale=1)
y2 = norm.pdf(x=x2, loc=5, scale=1)

y12 = np.convolve(y1, y2) / np.sum(y2)
x12 = np.linspace(min(x1) + min(x2), max(x1) + max(x2), len(y12))

y12_true = norm.pdf(x=x12, loc=0, scale=np.sqrt(1.25))

sns.lineplot(x=x12, y=y12)
#sns.lineplot(x=x12, y=y12_true)
plt.axvline(x=0, color='r', linestyle='--', label='Vertical Line at x=30')





# from scipy.stats import norm, binom
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.stats as stats
# from scipy import signal


# # uniform_dist = stats.norm(loc=-3, scale=4)
# # normal_dist = stats.norm(loc=3, scale=3)

# delta = 1e-4
# big_grid = np.arange(-15,15, delta)

# pmf1 = norm.pdf(big_grid, loc=-3, scale=4)*delta
# pmf2 = norm.pdf(big_grid, loc=3, scale=3)*delta

# conv_pmf = signal.fftconvolve(pmf1,pmf2,'same')
# np.convolve(pmf1,pmf2)
# print("Sum of convoluted pmf: "+str(sum(conv_pmf)))

# pdf1 = pmf1/delta
# pdf2 = pmf2/delta
# conv_pdf = conv_pmf/delta
# print("Integration of convoluted pdf: " + str(np.trapz(conv_pdf, big_grid)))


# plt.plot(big_grid,pdf1, label='Uniform')
# plt.plot(big_grid,pdf2, label='Gaussian')
# plt.plot(big_grid,conv_pdf, label='Sum')
# #plt.plot(big_grid,true_pdf, label='True')
# plt.legend(loc='best'), plt.suptitle('PDFs')
# plt.show()