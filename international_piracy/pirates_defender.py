import numpy as np
import json
from scipy.stats import beta, dirichlet


class Defender:
    def __init__(self, scaled = False):
        self.name = "Defender"
        self.scaled = scaled
        self.decisions = {
            "D1": [0, 1, 2, 3],  # Defender's decisions
            "A1": [0, 1, 2,3, 4],  # Attacker's decisions
            "D2": [0, 1, 2]}

        
        # Probabilities for Theta1
        self.p_theta1 = np.array([
            [0  , 0.4, 0, 0 , 0],  # d1^0
            [0  , 0.1, 0, 0 , 0],  # d1^1
            [0  , 0.05, 0, 0 , 0],   # d1^2
            [0  , 0,    0, 0 , 0]   # d1^3
        ])
        
                
        self.risk_aversion = 0.1
        
        
    def cost_function(self,d1, theta1, d2):
            
        cost_d1_d2_theta1_1 = np.array([
            [15.16, 2.3, 4.28],  ## d1^0
            [17.25, 4.39, 6.37], ## d1^1
            [19.39, 6.53, 8.51], ## d1^2
            [0.5,0.5,0.5]    ## d1^3
        ]) 
        
        cost_d1_d2_theta1_0 = np.array([
            [0, 0, 0],  ## d1^0
            [0.05, 0.05, 0.05], ## d1^1
            [0.15, 0.15, 0.15], ## d1^2
            [0.5,0.5,0.5]    ## d1^3
        ]) 
        
        
        if theta1 == 1:
            cost = cost_d1_d2_theta1_1[d1,d2]
        
        elif theta1 == 0:
            cost =  cost_d1_d2_theta1_0[d1,d2]
        
        return cost
        
        
    def utility_function(self, cost):


        return  -np.exp(self.risk_aversion * cost)
    

    def scaled_utility(self, cost):

        c_max = 19.36
        c_min = 0

        f_min = self.utility_function(c_max)  
        f_max = self.utility_function(c_min)  
    
        # Compute scaling parameters
        a = 1 / (f_max - f_min)
        b = - a * f_min
    
        #print(a)
        # Scaled function
        f_scaled = a * self.utility_function(cost) + b
        return f_scaled + 1
    
    def compute_utility(self, d1, d2, theta1):
        cost = self.cost_function(d1, theta1, d2)
        if self.scaled == True:
            ut =  self.scaled_utility(cost)
        else:
            ut = self.utility_function(cost)
        return ut
    
    
    def sample_theta1(self, d1, a1):
        prob = self.p_theta1[d1, a1]
        return np.random.choice([0, 1], p=[1 - prob, prob])

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
        
    def load_forescast(self, path_file):
        
        with open(path_file, 'r') as json_file:
            forescast = json.load(json_file)
        forescast = {int(i):forescast[i] for i in forescast.keys()}
        return forescast
    
    def d2_opt(self,d1, theta1):
        u_vec = np.zeros_like(self.decisions['D2'],dtype = 'float')
        for d2_ in self.decisions['D2']:
            u_vec[d2_] = self.compute_utility(d1 = d1, d2 = d2_, theta1 = theta1)
        return np.argmax(u_vec)
            
            
if __name__ == "__main__":
    
    defender = Defender()

    # Example decisions and outcomes
    d1_choice = 1  # Corresponds to d1^2
    a1_choice = 1  # Corresponds to a1^2
    d2_choice = 2  # Corresponds to d2^3

    # Sample Theta1 and Theta2
    sampled_theta1 = defender.sample_theta1(d1_choice, a1_choice)

    print(f"Sampled Theta1: {sampled_theta1}")

    # Compute utilities based on sampled Theta values
    defender_utility = defender.compute_utility(d1_choice, d2_choice, sampled_theta1)

    print(f"Defender Utility: {defender_utility}")


        
        


        