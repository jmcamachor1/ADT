from attacker_air_combat import Attacker
import numpy as np
from scipy.stats import mode
from tqdm import tqdm_notebook as tqdm

class Attacker_APS_fixed_h:
    
    def __init__(self, h = 1, N1 = 100, N2 = 100, burn = 0.75):
        self.name = "Attacker_APS_fixed_h"
        self.attacker = Attacker()
        self.h = h
        self.N1 = N1
        self.N2 = N2
        self.burn = burn
    
    
    
    def A2_given_d1_theta1_a1_APS(self,d1,a1,theta1, init_val = 'r'):
        
        prob_arr = np.zeros_like(self.attacker.decisions["A2"], dtype=float) 
        
        for n1 in range(self.N1):

            self.attacker.initialize_beliefs()
            a2_samples = np.zeros(self.N2, dtype=int)
            
            v = 0
            if init_val == 'r':
                a2_samples[0] = np.random.choice(self.attacker.decisions["A2"])
            else:    
                a2_samples[0] = init_val

            for i in range(self.h):
                
                d2_ = self.attacker.sample_d2(d1 = d1, 
                                              a1 = a1, 
                                              theta1 = theta1)
                
                
                theta2_ = self.attacker.sample_theta2(d2 = d2_, 
                                                      a2 = a2_samples[0], 
                                                      theta1 = theta1)
                
                
                v += np.log(self.attacker.compute_utility(d1 = d1, 
                                                          a1 = a1, 
                                                          theta1  =theta1, 
                                                         d2 = d2_, 
                                                        a2 = a2_samples[0], 
                                                        theta2 = theta2_)) 

            for i in range(1, self.N2):
                
                a2_tilde = self.attacker.propose(x_given = a2_samples[i-1],
                                                 x_values= self.attacker.decisions["A2"])
            
                
                v_tilde = 0
                
                for j in range(self.h):
                    
                    d2_tilde = self.attacker.sample_d2(d1 = d1, 
                                                          a1 = a1, 
                                                          theta1 = theta1)
                    
                    
                    theta2_tilde = self.attacker.sample_theta2(d2 = d2_tilde, 
                                                              a2 = a2_tilde, 
                                                              theta1 = theta1)
                    
                    v_tilde += np.log(self.attacker.compute_utility(d1 = d1, 
                                                                  a1 = a1, 
                                                                  theta1 = theta1, 
                                                                 d2 = d2_tilde, 
                                                                a2 = a2_tilde, 
                                                                theta2 = theta2_tilde))
                    
                alpha = np.exp(v_tilde - v)
                if np.random.uniform() < alpha:
                    v = v_tilde
                    a2_samples[i] = a2_tilde
                else:
                    a2_samples[i] = a2_samples[i-1]
                    v = v
                    
                
            idx = mode(a2_samples[int(self.burn*self.N2):])[0][0]
            prob_arr[idx]+=1/self.N1
            
        return prob_arr
    
    
        
    def A2_dist_APS(self, init_val = 'r'):
    
        A2_d1_theta1_a1 = {}
        
        for d1_ in self.attacker.decisions['D1']:
            a1_d = {}
            for a1_ in tqdm(self.attacker.decisions['A1']):
                theta1_d = {}
                for theta1_ in self.attacker.chance['Theta1']:
                    theta1_d[theta1_] = self.A2_given_d1_theta1_a1_APS(d1 = d1_,
                                                        a1 = a1_,
                                                        theta1 = theta1_, init_val = init_val)
                
                a1_d[a1_] = theta1_d
            A2_d1_theta1_a1[d1_] = a1_d
            
        return A2_d1_theta1_a1

    
    def A1_dist_APS(self, A2_d1_theta1_a1, init_val = 'r'):
        
        self.attacker.A2_d1_a1_theta1 = A2_d1_theta1_a1
        
        prob_arr = np.zeros_like(self.attacker.decisions["A1"], dtype=float)
        
        for n1 in range(self.N1):
            self.attacker.initialize_beliefs()
            a1_samples =  np.zeros(self.N2, dtype=int)
            
            if init_val == 'r':
                a1_samples[0] = np.random.choice(self.attacker.decisions["A1"])
                
            else:    
                a1_samples[0] = init_val
        
        
            v = 0
            
            for i in range(self.h):
                d1_  = self.attacker.sample_d1()
                theta1_ = self.attacker.sample_theta1(d1 = d1_, a1 = a1_samples[0])
                d2_ = self.attacker.sample_d2(d1 = d1_ , a1 = a1_samples[0], theta1 = theta1_ )
                p_A2 = self.attacker.A2_d1_a1_theta1[d1_][a1_samples[0]][theta1_]
                a2_ = np.random.choice(self.attacker.decisions["A2"], p = p_A2)            
                theta2_ = self.attacker.sample_theta2(d2 = d2_ , a2 =  a2_, theta1 = theta1_)
                v += np.log(self.attacker.compute_utility(d1 = d1_, a1 = a1_samples[0], theta1  =theta1_, 
                                                    d2 = d2_, a2 =a2_, theta2 = theta2_))
                
            for i in range(1, self.N2):
                
                a1_tilde = self.attacker.propose(x_given = a1_samples[i-1] ,x_values= self.attacker.decisions["A1"])
                v_tilde = 0
                
                for j in range(self.h):
                    d1_tilde = self.attacker.sample_d1()
                    theta1_tilde = self.attacker.sample_theta1(d1 = d1_tilde, a1 = a1_tilde)
                    #print('d1',d1_tilde, 'theta1', theta1_tilde, 'a1', a1_tilde)
                    d2_tilde = self.attacker.sample_d2(d1 = d1_tilde, a1 = a1_tilde, theta1 = theta1_tilde)
                    p_A2 = self.attacker.A2_d1_a1_theta1[d1_tilde][a1_tilde][theta1_tilde]
                    a2_tilde = np.random.choice(self.attacker.decisions["A2"], p = p_A2)
                    theta2_tilde = self.attacker.sample_theta2(d2 = d2_tilde, a2 = a2_tilde, theta1 = theta1_tilde)
                    v_tilde += np.log(self.attacker.compute_utility(d1 = d1_tilde, 
                                                                    a1 = a1_tilde, 
                                                                    theta1  =theta1_tilde, 
                                                        d2 = d2_tilde,
                                                        a2 =a2_tilde,
                                                        theta2 = theta2_tilde)) 

                alpha = np.exp(v_tilde - v)
                if np.random.uniform() < alpha:
                    v = v_tilde
                    a1_samples[i] = a1_tilde
                else:
                    a1_samples[i] = a1_samples[i-1]
                    v = v
                
            idx = mode(a1_samples[int(self.burn*self.N2):])[0][0]
            prob_arr[idx] += 1/self.N1
                
                
        return prob_arr 
                
            