# Sums of exotic distributions
This project converns a project of, for example, an insurance, where beforehand, it is unknown how many individuals will generate costs. And the costs themselves are also very unpredictable. In such contexts, densities of costs can be unpredictable. However, using the right statistical techniques, one can set boundaries of what total costs are reasonable.

In this setting, a random number of individuals will generate costs. Say this unknown number is modeled by $N~Binomial(n,p)$ where we speak of $n$ individuals each with a probability $p$ of generating costs. There is also a costs distribution $X$. The question an insurer could ask is, what is the probability of total costs of €1.500.000? Mathematically this is asking 
$$Pr\left(\sum_{i=1}^NX_i\leq z\right)$$
where $X_i$ follows a very unpredictable costs distribution. Using the fact typically many individuals generate costs, different statistical techniques because useful. The probability can be well approximated by first applying the _law of total probability_ and the _central limit theorem_. The latter tells us that, for fixed $k$, under mild conditions, the probability $\sum_{i=1}^kX_i\approx Normal(k\mu, k\sigma^2)$. Here $\mu$ and $\sigma$ are the mean and standard deviation of $X_i$.

The idea behind the solution is the following:
$$Pr\left(\sum_{i=1}^NX_i\leq z\right) = \sum_{i=1}^NPr\left(\sum_{i=1}^kX_i\leq z\right)Pr(N=k) \approx \sum_{i=1}^N\Phi(z; k\mu, k\sigma^2)Pr(N=k)$$
# Problem sketch

# Solution

# Ideas
