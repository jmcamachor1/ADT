import numpy as np
from scipy.stats import beta, dirichlet
from utils import beta1_dist,beta0_dist,dir001_dist,dir100_dist

class Attacker:
    def __init__(self):
        self.name = "Attacker"
        self.decisions = {
            "D1": [0, 1, 2],  # Defender's decisions
            "A1": [0, 1, 2],  # Attacker's decisions
            "D2": [0, 1, 2],
            "A2": [0, 1, 2]
        }

        self.chance = {
            "Theta1": [0,1],
            "Theta2":[0,1]
        }

        # Monetary value functions for decisions and uncertainties
        self.f_A_d1 = np.array([0.0, 0.4, 0.90])  # d1^1, d1^2, d1^3
        self.f_A_d2 = np.array([0.0, 0.45, 0.7])  # d2^1, d2^2, d2^3
        self.f_A_a1 = np.array([0.0, 0.2, 0.5])  # a1^1, a1^2, a1^3
        self.f_A_a2 = np.array([0.0, 0.85, 1])  # a2^1, a2^2, a2^3
        self.f_A_theta1 = np.array([0, 1.5])   # Theta1=0, Theta1=1
        self.f_A_theta2 = np.array([0, 3.5])   # Theta2=0, Theta2=1

        # Beta parameters for beliefs about Theta1 and Theta2
        self.beta_params_theta1 = np.array([
            [beta0_dist, beta(50, 50), beta(85, 15)],  # d1^0
            [beta0_dist, beta(20, 80), beta(30, 70)],  # d1^1
            [beta0_dist, beta(10, 90), beta(25, 75)]   # d1^2
        ])

        self.beta_params_theta2_given_theta1_1 = np.array([
            [beta0_dist,   beta1_dist, beta1_dist],  #d2^0
            [beta0_dist, beta(70, 30),beta(90, 10)],  # d2^1
            [beta0_dist, beta(50, 50), beta(85, 15)]   # d2^2
        ])

        self.beta_params_theta2_given_theta1_0 = np.array([
            [beta0_dist, beta(5, 95), beta(10, 90)],  # d2^0
            [beta0_dist, beta(2, 98), beta(8, 92)],  # d2^1
            [beta0_dist, beta(1, 99), beta(5, 95)]   # d2^2
        ])

        # Dirichlet parameters for beliefs about d2 given d1, a1, theta1
        self.dirichlet_params_d2_given_d1_a1_theta1_1 = np.array([
            [dirichlet([1, 999, 9000]), dirichlet([1, 499, 9500]), dirichlet([1, 999, 9000])],  # d1^0
            [dirichlet([1, 399, 9600]), dirichlet([1, 299, 9700]), dirichlet([1, 399, 9600])],  # d1^1
            [dirichlet([1, 199, 9800]), dirichlet([1, 99, 9900]), dirichlet([1, 199, 9800])]    # d1^2
        ])

        self.dirichlet_params_d2_given_d1_a1_theta1_0 = np.array([
            [dirichlet([1000, 1000, 1000]), dirichlet([1, 4499, 4500]), dirichlet([1, 1999, 8000])],  # d1^0
            [dirichlet([5000, 2500, 2500]), dirichlet([3500, 3500, 2000]), dirichlet([1, 1399, 8600])],  # d1^1
            [dirichlet([7500, 1500, 1000]), dirichlet([2000, 3000, 8000]), dirichlet([1, 1199, 8800])]    # d1^2
        ])

    def initialize_beliefs(self):
        # Sample risk aversion
        self.risk_proneness = np.random.uniform(1, 2)

        # Sample beliefs about d1
        # alpha = k * np.array([0.1, 0.2, 0,7])
        self.belief_d1 = dirichlet([1, 7500, 2499]).rvs()[0]

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

        # Sample beliefs about theta2 given theta1
        self.beliefs_theta2_given_theta1_1 = {}
        self.beliefs_theta2_given_theta1_0 = {}
        for d2 in self.decisions["D2"]:
            for a2 in self.decisions["A2"]:
                self.beliefs_theta2_given_theta1_1[(d2, a2)] = self.beta_params_theta2_given_theta1_1[d2, a2].rvs()
                
                self.beliefs_theta2_given_theta1_0[(d2, a2)] = self.beta_params_theta2_given_theta1_0[d2, a2].rvs()

    def cost_function(self, d1, a1, theta1, d2, a2, theta2):
        cost = (self.f_A_theta2[theta2] + self.f_A_theta1[theta1] - 
                (1 + self.f_A_d1[d1]) * self.f_A_a1[a1] -
                (1 + self.f_A_d2[d2]) * self.f_A_a2[a2])
        return cost

    def utility_function(self, cost):
        return np.exp( self.risk_proneness * cost)

    def compute_utility(self, d1, a1, theta1, d2, a2, theta2):
        cost = self.cost_function(d1, a1, theta1, d2, a2, theta2)
        return self.utility_function(cost)

    def sample_d1(self):
        return np.random.choice(self.decisions["D1"], p=self.belief_d1)
        
    def sample_theta1(self, d1, a1):
        belief = self.beliefs_theta1[(d1, a1)]
        return np.random.choice([0, 1], p=[1 - belief, belief])

    def sample_theta2(self, d2, a2, theta1):
        if theta1 == 1:
            belief = self.beliefs_theta2_given_theta1_1[(d2, a2)]
        else:
            belief = self.beliefs_theta2_given_theta1_0[(d2, a2)]
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
    a2_choice = 2  # Corresponds to a2^2

    # Sample beliefs and then Theta1 and Theta2
    sampled_theta1 = attacker.sample_theta1(d1_choice, a1_choice)
    sampled_theta2 = attacker.sample_theta2(d2_choice, a2_choice, sampled_theta1)

    print(f"Sampled Theta1: {sampled_theta1}")
    print(f"Sampled Theta2: {sampled_theta2}")

    # Sample d2 given d1, a1, theta1
    sampled_d2 = attacker.sample_d2(d1_choice, a1_choice, sampled_theta1)

    print(f"Sampled d2: {sampled_d2}")

    # Compute utilities based on sampled Theta values
    attacker_utility = attacker.compute_utility(d1_choice, a1_choice, sampled_theta1, d2_choice, a2_choice, sampled_theta2)

    print(f"Attacker Utility: {attacker_utility}")
