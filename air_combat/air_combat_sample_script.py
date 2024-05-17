import time, json, os
from sim_dm_problem_func import *


### params 
n_iters = ###
N_inner_ite = ####
n_proc = ####
dir_save = 'results/' 


for c_ in []:
    
    start = time.time()

    uD_with_fix_c = partial(uD  ,c = c_)


    A2_star_theta1_d1_a1  = compute_A2_theta1_d1_a1(iters = n_iters,
                                d1_values = [1,2,3],
                                theta1_values = [0,1],
                                a1_values = [1,2,3],
                                a2_values = [1,2,3],
                                N_inner_= N_inner_ite, 
                                burnin_=0.75, 
                                n_jobs = n_proc)

    params_D_theta2 = pDtheta2()
    theta2_sample_D = partial(theta2_sample, params = params_D_theta2)

    D2_star_theta1_d1_a1 = compute_APS_D2_theta1_d1_a1_parallel(d2_values = [1,2,3], 
                                  a2_values = [1,2,3],
                                  theta1_support = [0,1],
                                    d1_support = [1,2,3], 
                                    a1_support = [1,2,3],
                                    d_util = uD_with_fix_c,
                                    A2_theta1_d1_a1 = A2_star_theta1_d1_a1,
                                    theta2 = theta2_sample_D,
                                    N = N_inner_ite,
                                    burning = 0.75,
                                    n_jobs_ = n_proc)

    A1_star = compute_A1(iters =n_iters,
               a1_values = [1,2,3], 
               a2_values = [1,2,3], 
               A2_theta1_d1_a1 = A2_star_theta1_d1_a1, 
               N_inner_= N_inner_ite, 
             burnin_=0.75, 
            n_jobs = n_proc)


    params_D_theta1 = pDtheta1()
    theta1_sample_D = partial(theta1_sample ,params = params_D_theta1)

    D1_star = inner_APS_D1(d1_values = [1,2,3],
                 a1_values = [1,2,3],
                 a2_values = [1,2,3], 
                 A1 = A1_star, 
                 theta1 = theta1_sample_D,
                 theta2 = theta2_sample_D,
                 d2_opt = D2_star_theta1_d1_a1, 
                 A2_theta1_d1_a1 = A2_star_theta1_d1_a1, 
                 d_util = uD_with_fix_c, 
                 N = N_inner_ite, 
                 burnin = 0.75)


    exp_file = '_c_'+str(c_)+'_iters_'+str(n_iters)+'_N_inner_iters_'+str(N_inner_ite)

    with open(dir_save+'A2_star_theta1_d1_a1'+exp_file+'.json', 'w') as f:
        json.dump(str(A2_star_theta1_d1_a1), f)

    with open(dir_save + 'D2_star_theta1_d1_a1'+exp_file+'.json', 'w') as f:
        json.dump(str(D2_star_theta1_d1_a1), f)
    
    with open(dir_save + 'A1_star'+exp_file+'.json', 'w') as f:
        json.dump(str(A1_star), f)
    
    with open(dir_save+'D1_star'+exp_file+'.json', 'w') as f:
        json.dump(str(D1_star), f)

        
        
    end = time.time()

    print(exp_file,',TIME:',end-start)



