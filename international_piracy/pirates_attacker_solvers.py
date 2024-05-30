from pirates_attacker import Attacker
from scipy.stats import mode
import numpy as np
from tqdm import tqdm_notebook as tqdm


class Attacker_APS_fixed_h:
    
    def __init__(self, h  = 1 ,N1  = 100, N2 = 100, burn = 0.75):
        self.name = "Attacker_APS_fixed_h"
        self.attacker = Attacker()
        self.h = h
        self.N1 = N1
        self.N2 = N2
        self.burn = burn
    
    def A1_given_d1_MC(self,d1,init_val = 'r'):
        
        prob_arr = np.zeros_like(self.attacker.decisions["A1"], dtype=float)
        
        for n1 in tqdm(range(self.N1)):

            self.attacker.initialize_beliefs()
            a1_samples = np.zeros(self.N2, dtype=int)

            # Initialize attacker's decision
            if init_val == 'r':
                a1_samples[0] = np.random.choice(self.attacker.decisions["A1"])
            else:    
                a1_samples[0] = init_val
                
                
            v = 0
            for i in range(self.h):
                theta1_ = self.attacker.sample_theta1(d1 = d1, a1 = a1_samples[0])
                d2_ = self.attacker.sample_d2(d1 = d1, a1 = a1_samples[0], theta1 = theta1_)
                v += np.log(self.attacker.compute_utility(a1 = a1_samples[0], theta1 = theta1_, d2 = d2_, d1 = d1)) 


            for i in range(1, self.N2):

                a1_tilde = self.attacker.propose(x_given = a1_samples[i-1] ,x_values= self.attacker.decisions["A1"])
                v_tilde = 0

                for j in range(self.h):
                    theta1_tilde = self.attacker.sample_theta1(d1 = d1, a1 = a1_tilde)
                    d2_tilde = self.attacker.sample_d2(d1 = d1, a1 = a1_tilde, theta1 = theta1_tilde)
                    v_tilde += np.log( self.attacker.compute_utility(a1 = a1_tilde, theta1 = theta1_tilde, d2 = d2_tilde, d1 = d1) ) 

                alpha = np.exp(v_tilde - v)
                if np.random.uniform() < alpha:
                    v = v_tilde
                    a1_samples[i] = a1_tilde
                else:
                    a1_samples[i] = a1_samples[i-1]
                    v = v

            idx = mode(a1_samples[int(self.burn*self.N2):])[0][0]
            prob_arr[idx]+=1/self.N1
            
        return prob_arr
    
    def A1_dist_APS(self):
        
        A1_d1 = {}
        for d1 in list(self.attacker.decisions["D1"]):
            A1_d1[d1] = self.A1_given_d1_MC(d1)
        
        return A1_d1 
    
    
    
    
    
    

        
        
        

    

        
