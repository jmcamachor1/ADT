from defender_air_combat import Defender
import numpy as np
from scipy.stats import mode
from collections import Counter

    
class Defender_APS_fixed_h():
    
    def __init__(self, A1,A2_d1_theta1_a1, h  =1, N = 10, burn = 0.75):
        self.name = 'Defender_APS_fixed_h'
        self.defender = Defender()
        self.h = h
        self.N = N
        self.burn = burn
        self.A1 = A1
        self.A2_d1_theta1_a1 = A2_d1_theta1_a1
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

        
    def d2_given_d1_theta1_a1_APS(self,d1,a1,theta1, init_val = 'r'):
        
        
        d2_samples = np.zeros(self.N, dtype=int)
        
        if init_val == 'r':
            d2_samples[0] = np.random.choice(self.defender.decisions["D2"])
        else:
            d2_samples[0] = init_val
        
        v = 0
        for i in range(self.h):
            pa2_ = self.A2_d1_theta1_a1[d1][a1][theta1]
            a2_ = np.random.choice(self.defender.decisions["A2"], p = pa2_)
            theta2_ = self.defender.sample_theta2(d2 = d2_samples[0], a2 = a2_, theta1 = theta1)
            v+=np.log(self.defender.compute_utility(d1 = d1, a1 = a1, theta1  =theta1, 
                                            d2 = d2_samples[0], a2 = a2_, theta2 = theta2_))
        
        for i in range(1,self.N):
            
            d2_tilde = self.defender.propose(x_given = d2_samples[i-1],
                                             x_values = self.defender.decisions["D2"])
            
            v_tilde = 0
            
            for j in range(self.h):
                
                pa2_ = self.A2_d1_theta1_a1[d1][a1][theta1]
                a2_ = np.random.choice(self.defender.decisions["A2"], p = pa2_)
                theta2_ = self.defender.sample_theta2(d2 = d2_tilde, a2 = a2_, theta1 = theta1)
                v_tilde+=np.log(self.defender.compute_utility(d1 = d1, a1 = a1, theta1  =theta1, 
                                            d2 = d2_tilde, a2 = a2_, theta2 = theta2_))
                
            alpha = np.exp(v_tilde - v)
            if np.random.uniform() < alpha:
                d2_samples[i] = d2_tilde
                v = v_tilde
            else:
                d2_samples[i] = d2_samples[i-1]
                
        d2_opt = mode(d2_samples[int(self.burn*self.N):])[0][0]
        d2_dist = d2_samples[int(self.burn*self.N):]

        return d2_opt, d2_dist, Counter(d2_dist)            
            
        
    def D2_star_dist_APS(self):

        imp_list = [(0,0,1),(1,0,1),(2,0,1)]
    
        d2_d1_theta1_a1 = {}
        
        for d1_ in self.defender.decisions['D1']:
            a1_d = {}
            for a1_ in self.defender.decisions['A1']:
                theta1_d = {}
                for theta1_ in self.defender.chance['Theta1']:
                    if (d1_,a1_,theta1_) not in imp_list:
                        theta1_d[theta1_] = self.d2_given_d1_theta1_a1_APS(d1 = d1_,
                                                        a1 = a1_,
                                                        theta1 = theta1_)[0]
                    else:
                        theta1_d[theta1_] = -1
                                            

                
                a1_d[a1_] = theta1_d
            d2_d1_theta1_a1[d1_] = a1_d
            
        return d2_d1_theta1_a1



    def D1_star_dist_APS(self, D2_d1_a1_theta1, init_val = 'r'):
        
        d1_samples =  np.zeros(self.N, dtype=int)
        if init_val == 'r':
            d1_samples[0] = np.random.choice(self.defender.decisions["D1"])
        else:
            d1_samples[0] = init_val
        
        v = 0
        
        for i in range(self.h):
            a1_ = np.random.choice(self.defender.decisions["A1"], 
                                   p = self.A1)
            
            theta1_ = self.defender.sample_theta1(d1 = d1_samples[0], 
                                                  a1 = a1_)
            
            d2_ = D2_d1_a1_theta1[d1_samples[0]][a1_][theta1_]
            pA2 = self.A2_d1_theta1_a1[d1_samples[0]][a1_][theta1_]
            a2_ = np.random.choice(self.defender.decisions["A2"], p= pA2)
            theta2_ = self.defender.sample_theta2(d2 = d2_, a2 = a2_, theta1 = theta1_)
            v+=np.log(self.defender.compute_utility(d1 = d1_samples[0], a1 = a1_, 
                                                    theta1  =theta1_, 
                                                    d2 = d2_, 
                                                    a2 = a2_, theta2 = theta2_))
        for i in range(1,self.N):

            d1_tilde = self.defender.propose(x_given = d1_samples[i-1],
                                                x_values = self.defender.decisions["D1"])
            v_tilde = 0
            
            for j in range(self.h):
                a1_tilde = np.random.choice(self.defender.decisions["A1"], p = self.A1)
                theta1_tilde = self.defender.sample_theta1(d1 = d1_tilde, a1 = a1_tilde)
                a2_tilde = np.random.choice(self.defender.decisions["A2"], p = self.A2_d1_theta1_a1[d1_tilde][a1_tilde][theta1_tilde])
                d2_tilde = D2_d1_a1_theta1[d1_tilde][a1_tilde][theta1_tilde]
                theta2_tilde = self.defender.sample_theta2(d2 = d2_tilde, a2 = a2_tilde, theta1 = theta1_tilde)
                
                
                v_tilde+=np.log(self.defender.compute_utility(d1 = d1_tilde, a1 = a1_tilde, theta1  =theta1_tilde, 
                                                d2 = d2_tilde, a2 = a2_tilde, theta2 = theta2_tilde))
                
            alpha = np.exp(v_tilde - v)
            if np.random.uniform() < alpha:
                d1_samples[i] = d1_tilde
                v = v_tilde
            else:
                d1_samples[i] = d1_samples[i-1]
                    
        d1_opt = mode(d1_samples[int(self.burn*self.N):])[0][0]
        d1_dist = d1_samples[int(self.burn*self.N):]
        return  d1_opt, d1_samples,Counter(d1_dist) 
    
    