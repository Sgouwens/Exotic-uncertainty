import numpy as np
import pandas as pd
import seaborn as sns
from distmaker import Distributionfinder, Convolver

from scipy.stats import norm
# Implement actual use case where we have a whole population (N_pop varies over age, and p too)
# Then use Convolver to convolute it together

p = 0.1
N_pop = 100000
N_cost = int(np.ceil(N_pop*p))

price = np.random.normal(loc=250, scale=10, size=N_cost)
amount = np.random.negative_binomial(9, 0.85, N_cost)
costs = price*amount
costs = costs - np.mean(costs)

distf = Distributionfinder(data=costs, popsize=N_pop, prob=p)

# distf.generate_bootstrap_density(5000)
# distf.plot_density(type='bootstrap')

distf.generate_dist_clt_trick(-2e5, 2e5, 500)
#distf.plot_density(type='clt')

# Convolver zodanig aanpassen dat een element van Distfinder gewoon alles mooi laat lopen. geen lelijke dingen met [1]
convtest = Convolver(distf.density_clt[0], distf.density_clt[1])
convtest2 = convtest + convtest
convtest2.plot_density()

# Als convolueren altijd goed gaat zonder onnodig lange intervals
# def prob_fun(x):
#     return (0.3 - 0.027*x + 0.02*(x/5)**2 - 0.05*(x/20)**3) / 2

# a_range = np.linspace(0,80, num=81)
# p_range = prob_fun(a_range)

# sns.lineplot(x=a_range, y=p_range)
