import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import norm, binom, gaussian_kde
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

import matplotlib.pyplot as plt

p = 0.1
N_pop = 100000
N_cost = int(np.ceil(N_pop*p))

price = np.random.normal(loc=250, scale=10, size=N_cost)
amount = np.random.negative_binomial(9, 0.85, N_cost)
costs = price*amount

class Distributionfinder:
    
    costs_mean = np.mean(costs)
    costs_sd= np.std(costs)    
    
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
        # First we generate bootstrapped samples, then we use KDE to compute density
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
        
        costs_mean = np.mean(costs)
        costs_sd = np.std(costs)  
        
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
    Given densities of X and Y, this computes the distribution of X+Y
    
    This class should not be used at this point"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        pass
        """Fourier transform for convolution is much faster than using numpy's
        convolve function which performs numerical integration. The accuracy in
        our use case is equal. However in general this may not be true."""
        
        common_grid = np.linspace(min(self.x.min(), other.x.min()), max(self.x.max(), other.x.max()), 300)
        delta_x = common_grid[1] - common_grid[0]
        
        y1_on_common_grid = interp1d(self.x, self.y, kind='linear', fill_value=0, bounds_error=False)(common_grid)
        y2_on_common_grid = interp1d(other.x, other.y, kind='linear', fill_value=0, bounds_error=False)(common_grid)
        
        convolution_result = fftconvolve(y1_on_common_grid, y2_on_common_grid, mode='same') * delta_x
        
        return Convolver(common_grid, convolution_result)
    
    def get_quantiles(self, q1, q2):
        # Use interpolation to do this accurately
        pass
        
distf = Distributionfinder(data=costs, popsize=N_pop, prob=p)

# distf.generate_bootstrap_density(5000)
# distf.plot_density(type='bootstrap')

distf.generate_dist_clt_trick(3.72e6, 4.2e6, 500)
#distf.plot_density(type='clt')


convtest = Convolver(distf.density_clt[0], distf.density_clt[1])

plt.plot(convtest.x, convtest.y)

convtest2 = convtest + convtest

plt.plot(convtest2.x, convtest2.y)