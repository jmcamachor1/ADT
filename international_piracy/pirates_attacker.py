import numpy as np
from scipy.stats import beta, dirichlet
from utils import beta1_dist, beta0_dist, dir001_dist, dir100_dist




class Attacker:
    def __init__(self):
        self.name = "Attacker"
        self.decisions = {
            "D1": [0, 1, 2, 3],  # Defender's decisions
            "A1": [0, 1, 2,3, 4],  # Attacker's decisions
            "D2": [0, 1, 2]}
        
                # Beta parameters for beliefs about Theta1 and Theta2
        self.beta_params_theta1 = np.array([
            [beta0_dist, beta(40, 60),    beta(1, 1), beta(1,1), beta(1,1)],   # d1^0
            [beta0_dist, beta(10, 90),    beta(1, 1), beta(1,1), beta(1,1)],  # d1^1
            [beta0_dist, beta(50, 950),   beta(1, 1), beta(1,1),beta(1,1)],    # d1^2
            [beta0_dist, beta0_dist, beta(1, 1), beta(1, 1),beta(1, 1)]    # d1^3
        ])

        self.dirichlet_params_d2_given_d1_a1_theta1_1  = np.array([
            [dir100_dist, dirichlet([1,1,1]),    dirichlet([1, 1,1]), dirichlet([1,1,1]), dirichlet([1,1,1])],   # d1^0
            [dir100_dist, dirichlet([0.1,4,6]),    dirichlet([1, 1,1]), dirichlet([1,1,1]), dirichlet([1,1,1])],  # d1^1
            [dir100_dist, dirichlet([0.1,1,10]),   dirichlet([1, 1,1]), dirichlet([1,1,1]), dirichlet([1,1,1])],    # d1^2
            [dir100_dist, dir100_dist,      dir100_dist, dir100_dist, dir100_dist]    # d1^3
        ])
        
        self.dirichlet_params_d2_given_d1_a1_theta1_0  = np.array([
            [dir100_dist, dir100_dist,    dir100_dist, dir100_dist, dir100_dist],   # d1^0
            [dir100_dist, dir100_dist,    dir100_dist, dir100_dist, dir100_dist],  # d1^1
            [dir100_dist, dir100_dist,   dir100_dist, dir100_dist, dir100_dist],    # d1^2
            [dir100_dist, dir100_dist,      dir100_dist, dir100_dist, dir100_dist]    # d1^3
        ])
        
    
        
    def initialize_beliefs(self):
        
        # Sample risk proneness
        self.risk_prone = np.random.uniform(0, 20)
        
        # Sample beliefs about d2 given d1, a1, theta1
        self.beliefs_d2_given_d1_a1_theta1_1 = {}
        self.beliefs_d2_given_d1_a1_theta1_0 = {}

        for d1 in self.decisions["D1"]:
            for a1 in self.decisions["A1"]:
                self.beliefs_d2_given_d1_a1_theta1_1[(d1, a1)] = self.dirichlet_params_d2_given_d1_a1_theta1_1[d1, a1].rvs()
                self.beliefs_d2_given_d1_a1_theta1_0[(d1, a1)] = self.dirichlet_params_d2_given_d1_a1_theta1_0[d1, a1].rvs()
            
        # Sample beliefs about theta1
        self.beliefs_theta1 = {}
        for d1 in self.decisions["D1"]:
            for a1 in self.decisions["A1"]:
                self.beliefs_theta1[(d1, a1)] = self.beta_params_theta1[d1, a1].rvs()
                
    
    def cost_function(self, a1, theta1, d2, d1):

        costs_d2_a1_theta1 =  np.array([
            [0, 0.97, 0.97, 0.97, 0.97],   # d2^0
            [0, 2.27, 2.27, 2.27, 2.27],  # d2^1
            [0,  -1.28,  -1.28, -1.28, -1.28]    # d2^2
            ])
        
        costs_d2_a1_theta0 =  np.array([
            [0,-0.53, -0.53, -0.53, -0.53],   # d2^0
            [0,-0.53, -0.53, -0.53, -0.53],  # d2^1
            [0,-0.53, -0.53, -0.53, -0.53]    # d2^2
            ])
            
        if theta1 == 1:
            cost = costs_d2_a1_theta1[d2, a1]
        elif theta1 == 0:
            cost = costs_d2_a1_theta0[d2, a1]

        return cost            
    
    def utility_function(self, cost):
            
        return np.exp(self.risk_prone * cost) + 1

    def compute_utility(self, a1, theta1, d2, d1):
        cost = self.cost_function(a1, theta1, d2, d1)
        return self.utility_function(cost)
    
    def sample_theta1(self, d1, a1):
        belief = self.beliefs_theta1[(d1, a1)]
        return np.random.choice([0, 1], p=[1 - belief, belief])

    def sample_d2(self, d1, a1, theta1):
        if theta1 == 1:
            belief = self.beliefs_d2_given_d1_a1_theta1_1[(d1, a1)].squeeze()
        else:
            belief = self.beliefs_d2_given_d1_a1_theta1_0[(d1, a1)].squeeze()
        return np.random.choice(self.decisions["D2"], p=belief)
    
    def propose(self,x_given, x_values):

        ### SYMMETRIC FUNCTION SO IT CANCELS IN Metropolis Hastings

        if x_given == x_values[0]:
            return( np.random.choice([x_values[1], x_values[-1]],
            p=[0.5, 0.5]) )

        if x_given == x_values[-1]:
            return( np.random.choice([x_values[0], x_values[-2]],
            p=[0.5, 0.5]) )

        idx = list(x_values).index(x_given)
        return( np.random.choice([x_values[idx+1], x_values[idx-1]],
        p=[0.5, 0.5]) )





if __name__ == "__main__":
    
    # Example usage
    attacker = Attacker()
    attacker.initialize_beliefs()

    # Example decisions and outcomes
    d1_choice = 1  # Corresponds to d1^1
    a1_choice = 1  # Corresponds to a1^1
    d2_choice = 2  # Corresponds to d2^2

    # Sample beliefs and then Theta1 and Theta2
    sampled_theta1 = attacker.sample_theta1(d1_choice, a1_choice)

    print(f"Sampled Theta1: {sampled_theta1}")

    # Sample d2 given d1, a1, theta1
    sampled_d2 = attacker.sample_d2(d1_choice, a1_choice, sampled_theta1)

    print(f"Sampled d2: {sampled_d2}")

    # Compute utilities based on sampled Theta values
    attacker_utility = attacker.compute_utility(a1_choice, sampled_theta1, sampled_d2)

    print(f"Attacker Utility: {attacker_utility}")
