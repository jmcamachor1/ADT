import json, time
from seq_dm_problem_func import *

### params 
n_iters = #xxxx
N_inner_ite =#xxx
n_proc = 2#xxxxx
dir_save = '' 

for c_ in [0.1,0.4,1,2]:
    #### compute A_d
    start = time.time()
    A_d  = compute_A_d(iters = n_iters,
                       d_values = [1,2,3,4], 
                       a_values = [0,1,2,3,4],
                       N_inner_=N_inner_ite,
                       burnin_= 0.75, 
                       n_jobs = n_proc)

    uD_1 = partial(uD, c = c_)
    theta_params = pDtheta()
    theta_f = partial(theta_sample,params = theta_params)
    APS_D2_1 = partial(APS_D2, c = c_)
    r = inner_APS_D1(d_values = [1,2,3], 
                 a_values = [0,1,2,3,4], 
                 d_util = uD_1 ,
                 A_d = A_d, 
                 theta = theta_f, 
                 d2_opt = APS_D2_1,
                 N = N_inner_ite, 
                 burnin = 0.75)
    
    rd2 = {'d2':APS_D2_1(d1 = r['mode'][0], theta = 1)}
    
    path_to_save = dir_save+'_c_'+str(c_)+'_iters_'+str(n_iters)+'N_inner_iters_'+str(N_inner_ite)
    
    with open('d1'+path_to_save+'.json', 'w') as f:
        json.dump(str(r), f)
    
    with open('A_d'+path_to_save+'.json', 'w') as f:
        json.dump(str(A_d), f)
        
    with open('d2'+path_to_save+'.json', 'w') as f:
        json.dump(str(rd2), f)

    end = time.time()

    print(path_to_save,',TIME:',end-start)