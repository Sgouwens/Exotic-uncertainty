import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import norm, binom, gaussian_kde
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.optimize import bisect
from scipy.signal import fftconvolve

import matplotlib.pyplot as plt


class Distributionfinder:
    
    def __init__(self, data, popsize, prob, density_bootstrap=None, dist_clt_trick=None):
        self.data = data
        self.popsize = popsize
        self.prob = prob
        self.density_bootstrap = density_bootstrap
        self.density_clt = dist_clt_trick
        
    def single_bootstrap_sample(self):
        """"""
        amounts = np.random.binomial(self.popsize, self.prob, 1)
        prices = np.random.choice(self.data, size=amounts)
        return np.sum(prices)
    
    def generate_bootstrap_density(self, B):
        """"""
        bootstrapped_samples = np.array([self.single_bootstrap_sample() for _ in range(B)])
        bootstrapped_density = gaussian_kde(bootstrapped_samples, bw_method='silverman')

        x_values = np.linspace(min(bootstrapped_samples), max(bootstrapped_samples), 1000)
                
        self.density_bootstrap = np.array([x_values, bootstrapped_density(x_values)])
        
    def plot_density(self, type='clt'): # Maak hier een density van
        """"""
        if type == 'clt':
            points = self.density_clt
        if type == 'bootstrap':
            points = self.density_bootstrap
            
        sns.lineplot(x=points[0], y=points[1])
        
    def generate_dist_clt_value(self, z):
        """"""
        
        costs_mean = np.mean(self.data)
        costs_sd = np.std(self.data)  
        
        binom_mean = self.popsize * self.prob
        binom_sd = np.sqrt(self.popsize * self.prob * (1-self.prob))
        # Ensure the grid is wide enough to prevent numerical errors getting big
        int_lower = int(np.floor(binom_mean - 3*binom_sd))
        int_upper = int(np.ceil(binom_mean + 3*binom_sd))
        
        it = range(int_lower, int_upper)
    
        val = np.sum(binom.pmf(it, self.popsize, self.prob) * \
            norm.pdf(z, loc=np.array(it)*costs_mean, scale=np.sqrt(it)*costs_sd))
                
        return val

    def generate_dist_clt_trick(self, lower, upper, num_points = 50):
        """"""
        space = np.linspace(lower, upper, num_points)
        prob_costs = np.full(num_points, None)
        
        for s in range(num_points):
            prob_costs[s] = self.generate_dist_clt_value(space[s])
            
        self.density_clt = np.array([space, prob_costs])
    
        
class Convolver():
    """Input two densities of the class Distributionfinder for convolution. 
    Given densities of X and Y, this computes the distribution of X+Y"""
    
    def __init__(self, x, y):
        # make sure can only be loaded if dist is non-empty
        self.x = x
        self.y = y
        # Here, apply some useful functions beforehand.
    
    def __add__(self, other):
        pass
        """Fourier transform for convolution is much faster than using numpy's
        convolve function which performs numerical integration. The accuracy in
        our use case is equal. However in general this may not be true."""
        
        common_grid = np.linspace(min(self.x.min(), min(other.x))-1500, 
                                  max(self.x.max(), other.x.max())+2000, 
                                  20000)
        
        delta_x = common_grid[1] - common_grid[0]
        
        y1_on_common_grid = interp1d(self.x, self.y, kind='linear', fill_value=0, bounds_error=False)(common_grid)
        y2_on_common_grid = interp1d(other.x, other.y, kind='linear', fill_value=0, bounds_error=False)(common_grid)
        # Below does not work as it should yet
        
        convolution_result = fftconvolve(y1_on_common_grid, y2_on_common_grid, mode='same') * delta_x
        
        # normaliser = self.compute_moments(1)
        
        # new_distribution = Convolver(common_grid, convolution_result)
        # new_distribution.correct_boundaries()
        return Convolver(common_grid, convolution_result)
    
    def plot_density(self):
        sns.lineplot(x=self.x, y=self.y)
    
    def compute_moments(self, k):
        """Computes the k'th moment of the distribution using numerical integration"""
        return simps(self.x**k * self.y, self.x)
    
    def compute_probability(self, lower, upper): # Use the base of get_quantile
        """Computes the integral int_lower^upper f(x) dx"""
        minx = min(self.x)
        maxx = max(self.x)
        
        idx = (max(lower, minx) < self.x) & (min(upper, maxx) < self.x) # not correct
        return simps(self.y[idx], self.x[idx])
    
    def get_sd(self):
        return np.sqrt(self.compute_moments(2) - self.compute_moments(1)**2)
            
    def interpolate_pdf(self, value):
        dx = self.x[1] - self.x[0]
        interp_func = interp1d(self.x, self.y, kind='linear', fill_value='extrapolate')
        return float(interp_func(value))
    
    def get_quantiles(self, val):
        """WE NEED TO AGGREGATE HERE"""
        dx = self.x[1] - self.x[0]        
        cumsum_y = np.cumsum(self.y) * dx
         
        interp_func = interp1d(self.x, cumsum_y, kind='linear', fill_value='extrapolate')
        return float(interp_func(val))
    
    def get_probability(self, val1, val2):
        return self.get_quantiles(val2) - self.get_quantiles(val1)
            
    def get_ranges(self, bounds_at=1e-7, tol=1e-9):
        """This function may fail when using an unimodal function or when the variance is very large."""
        
        mean = self.compute_moments(1)
        sd = self.get_sd()
        
        def root_function(x):
            return self.interpolate_pdf(x) - bounds_at
        
        root_left = bisect(root_function, mean - 10*sd, mean, xtol=tol)
        root_right = bisect(root_function, mean + 10*sd, mean, xtol=tol)
        return root_left, root_right
        
    def correct_boundaries(self):
        """Iteratively search density function to find a value such that f is under 1e-6"""
        lower, upper = self.get_ranges()
        idx = (lower < self.x) & (self.x < upper)
        self.x = self.x[idx]
        self.y = self.y[idx]
        