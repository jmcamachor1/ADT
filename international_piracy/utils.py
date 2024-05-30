import scipy.stats as stats
import numpy as np



def create_deg_beta(v,p):
    # Define the degenerate distribution at x0
    values = ([v], [p])  # Only one point with probability 1

    return stats.rv_discrete(name='deg_prob1', values=values)

# Define the custom Degenerate Dirichlet distribution
class DegenerateDirichlet():
    def __init__(self, p):
        super().__init__()
        self.p = np.asarray(p)
    
    def rvs(self, size=None, random_state=None):
        size = 1 if size is None else size
        return np.tile(self.p, (size, 1))



beta1_dist = create_deg_beta(v = 1, p = 1)
beta0_dist = create_deg_beta(v = 0, p = 1)
dir001_dist = DegenerateDirichlet(p = [0,0,1])
dir100_dist = DegenerateDirichlet(p = [1,0,0])








