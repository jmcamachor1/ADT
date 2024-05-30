
from pirates_defender import Defender
import numpy as np
from collections import Counter
from scipy.stats import mode
            

class  Defender_APS_fixed_h:
    
    def __init__(self, A1_d1, h  = 1 ,N  = 100, burn = 0.75):
        self.name = "Defender_APS_fixed_h"
        self.defender = Defender()
        self.defender.A1_d1 = A1_d1
        self.h = h
        self.N = N
        self.burn = burn
    
    def D1_star(self, init_val = 'r'):
        
        d1_samples = np.zeros(self.N, dtype=int)
        
        if init_val == 'r':
            d1_samples[0] = np.random.choice(self.defender.decisions["D1"])
        else:
            d1_samples[0] = init_val
            
        v = 0

        for i in range(self.h):
            a1_ = np.random.choice(self.defender.decisions['A1'], 
                           p = self.defender.A1_d1[d1_samples[0]])
            theta1_ = self.defender.sample_theta1(d1 = d1_samples[0],a1 = a1_)
            d2_ = self.defender.d2_opt(d1 = d1_samples[0] , theta1 = theta1_)
            v += np.log(self.defender.compute_utility(d1 = d1_samples[0], 
                                                        d2 = d2_, 
                                                        theta1 = theta1_)) 
            
        
        for i in range(1,self.N):
            
            d1_tilde = self.defender.propose(x_given = d1_samples[i-1] ,
                                  x_values= self.defender.decisions["D1"])
            v_tilde = 0

            for j in range(self.h):
                a1_tilde = np.random.choice(self.defender.decisions['A1'],
                                            p = self.defender.A1_d1[d1_tilde])
                theta1_tilde = self.defender.sample_theta1(d1 = d1_tilde, 
                                                        a1 = a1_tilde)
                d2_tilde = self.defender.d2_opt(d1 = d1_tilde , 
                                            theta1 = theta1_tilde)
                v_tilde += np.log(self.defender.compute_utility(d1 = d1_tilde ,
                                                                    d2 = d2_tilde, 
                                                                    theta1 = theta1_tilde)) 

            alpha = np.exp(v_tilde - v)
            if np.random.uniform() < alpha:
                v = v_tilde
                d1_samples[i] = d1_tilde
            else:
                d1_samples[i] = d1_samples[i-1]
                v = v

                    
        d1_opt = mode(d1_samples[int(self.burn*self.N):])[0][0]
        d1_dist = d1_samples[int(self.burn*self.N):]
        return d1_opt,Counter(d1_dist)
        

                