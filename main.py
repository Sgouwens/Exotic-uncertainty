# TODO: Use FFT to add all distributions together to get combined distribution
# This can be approximated by normals, but that's boring.

# Figure out what are the requirements to the density to perform fft well.

import numpy as np
import pandas as pd
import seaborn as sns
from distmaker import Distributionfinder, Convolver

from scipy.stats import norm

def prob_fun(x):
    return (0.3 - 0.027*x + 0.02*(x/5)**2 - 0.05*(x/20)**3) / 2


ages = [5*k for k in range(2, 16)]

def generate_data(a, compare_bootstrap_and_clt=False, lowerbound=0, upperbound=4e5):
    
    p = prob_fun(a)
    
    N_pop = 20000 - a ** 2
    N_costs = int(np.ceil(N_pop*p))
    
    price = np.random.normal(loc=20+a/10, scale=10+a/40, size=N_costs)
    amount = np.random.negative_binomial(9, 0.7-a/500, N_costs)
    
    costs = price*amount
    
    distf = Distributionfinder(costs, N_pop, p)
    distf.generate_dist_clt_trick(lowerbound, upperbound, 1000)
    
    if compare_bootstrap_and_clt:
        distf.generate_bootstrap_density(5000)
        distf.plot_density(type='bootstrap')
        distf.plot_density(type='clt')
        
        return None
    
    convtest = Convolver(distf.density_clt[0], distf.density_clt[1])
    
    return convtest
    
def main():
    
    generate_data(80, compare_bootstrap_and_clt=True, lowerbound=6e4, upperbound=1e5)
    
    list_densties = [generate_data(a) for a in ages]
    
    for dens in list_densties:
        dens.plot_density()
        
if __name__=='__main__':
    main()
    

