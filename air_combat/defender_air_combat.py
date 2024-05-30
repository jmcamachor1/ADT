import numpy as np

class Defender:
    def __init__(self):
        self.name = "Defender"
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
        self.f_D_d1 = np.array([0.0, 0.3, 0.6])  # d1^1, d1^2, d1^3
        self.f_D_d2 = np.array([0.0, 0.3, 1.5])  # d2^1, d2^2, d2^3
        self.f_D_a1 = np.array([0.0, 0.4, 0.5])  # a1^1, a1^2, a1^3
        self.f_D_a2 = np.array([0.0, 0.5, 0.8])  # a2^1, a2^2, a2^3
        self.f_D_theta1 = np.array([0.0, 2.0])   # Theta1=0, Theta1=1
        self.f_D_theta2 = np.array([0.0, 7.5])   # Theta2=0, Theta2=1

        self.risk_aversion = 0.2
        self.scaled =  True

        # Probabilities for Theta1 and Theta2 as numpy matrices
        self.p_theta1 = np.array([
            [0.00, 0.70, 0.80],  # d1^1
            [0.00, 0.35, 0.45],  # d1^2
            [0.00, 0.30, 0.4]   # d1^3
        ])

        self.p_theta2_given_theta1_1 = np.array([
            [0.00, 0.85, 0.95],  # d2^1
            [0.00, 0.40, 0.60],  # d2^2
            [0.00, 0.05, 0.15]   # d2^3
        ])

        self.p_theta2_given_theta1_0 = np.array([
            [0.00, 0.25, 0.35],  # d2^1
            [0.00, 0.05, 0.07],  # d2^2
            [0.00, 0.01, 0.02]   # d2^3
        ])

    def cost_function(self, d1, a1, theta1, d2, a2, theta2):
        cost = (self.f_D_theta2[theta2] + self.f_D_theta1[theta1] + 
                (1 + self.f_D_a1[a1]) * self.f_D_d1[d1] + 
                (1 + self.f_D_a2[a2]) * self.f_D_d2[d2])
        return cost
    
    def scaled_utility(self, cost):

        c_max = (self.f_D_theta2[1] + self.f_D_theta1[1] + 
                (1 + self.f_D_a1[2]) * self.f_D_d1[2] + 
                (1 + self.f_D_a2[2]) * self.f_D_d2[2])
        c_min = (self.f_D_theta2[0] + self.f_D_theta1[0] + 
                (1 + self.f_D_a1[0]) * self.f_D_d1[0] + 
                (1 + self.f_D_a2[0]) * self.f_D_d2[0])

        f_min = self.utility_function(c_max)  
        f_max = self.utility_function(c_min)  
    
        # Compute scaling parameters
        a = 1 / (f_max - f_min)
        b = - a * f_min
    
        #print(a)
        # Scaled function
        f_scaled = a * self.utility_function(cost) + b
        return f_scaled + 1


    def utility_function(self, cost):
        return -np.exp(self.risk_aversion * cost) +1

    def compute_utility(self, d1, a1, theta1, d2, a2, theta2):

    #def compute_utility(self, d1, d2, theta1):
    #    cost = self.cost_function(d1, theta1, d2)
    #    if self.scaled == True:
    #        ut =  self.scaled_utility(cost)
    #    else:
    #        ut = self.utility_function(cost)
    #    return ut

        cost = self.cost_function(d1, a1, theta1, d2, a2, theta2)
        if self.scaled == True:
            ut = self.scaled_utility(cost)
        else:
            ut = self.utility_function(cost)

        return ut

    def sample_theta1(self, d1, a1):
        prob = self.p_theta1[d1, a1]
        return np.random.choice([0, 1], p=[1 - prob, prob])

    def sample_theta2(self, d2, a2, theta1):
        if theta1 == 1:
            prob = self.p_theta2_given_theta1_1[d2, a2]
        else:
            prob = self.p_theta2_given_theta1_0[d2, a2]
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


    



if __name__ == "__main__":

    # Example usage
    defender = Defender()

    # Example decisions and outcomes
    d1_choice = 1  # Corresponds to d1^2
    a1_choice = 1  # Corresponds to a1^2
    d2_choice = 2  # Corresponds to d2^3
    a2_choice = 2  # Corresponds to a2^3

    # Sample Theta1 and Theta2
    sampled_theta1 = defender.sample_theta1(d1_choice, a1_choice)
    sampled_theta2 = defender.sample_theta2(d2_choice, a2_choice, sampled_theta1)

    print(f"Sampled Theta1: {sampled_theta1}")
    print(f"Sampled Theta2: {sampled_theta2}")

    # Compute utilities based on sampled Theta values
    defender_utility = defender.compute_utility(d1_choice, a1_choice, sampled_theta1, d2_choice, a2_choice, sampled_theta2)

    print(f"Defender Utility: {defender_utility}")
