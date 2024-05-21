import numpy as np
from param_func_sim import cD, cA, UA_sample,pAd2_sample, pAd1_sample,pAtheta2_sample, pAtheta1_sample,pDtheta1, pDtheta2
from functools import partial
from scipy.stats import mode
from joblib import Parallel, delayed
import itertools




def uD(d1, d2, theta1,theta2,a1, a2,  c):    ### CHECK
    """
    Computes the Defender's utility depending on d1, d2, theta1,theta2,a1, a2,  

    Paramaters
    ----------
    d1: Integer
    1st Defender's decision
    d2: Integrer
    2nd Defender's decision
    theta1: Integer
    Variable indicating if attack is succeful to air anticraft
    theta2: Integer
    Variable indicating if attack is succeful to infrastructur
    a1: Integer
    1st Ataccker's decision
    a2: Integrer
    2nd Attacker's decision
    c: Float
    Risk aversion coefficient
    
    Returns
        Float: Utility value of Defender 

    """

    alpha_D = np.exp(c* 13) ## remain utility positive

    return - np.exp([c * cD(d1, d2, theta1,theta2,a1, a2)]) + alpha_D


def ua_sample(d1, d2, theta1,theta2,a1, a2,params):     #### CHECK
  
    """
    Computes the Attakcr's utility depending on a1, theta and d2

    Paramaters
    ----------
    d1: Integer
    1st Defender's decision
    d2: Integrer
    2nd Defender's decision
    theta1: Integer
    Variable indicating if attack is succeful to air anticraft
    theta2: Integer
    Variable indicating if attack is succeful to infrastructur
    a1: Integer
    1st Ataccker's decision
    a2: Integrer
    2nd Attacker's decision    params: Float
    Risk averse coefficient

    Returns
    Float: Utility value of a1, theta and d2

    """
  

    return np.exp([params * cA(d1, d2, theta1,theta2,a1, a2)])

def theta1_sample(d1,a1, params):     ##### CHECK
    """
    Sample from Theta1 regarding  d1, a1 using params

    Paramaters
    ----------
    d1: Integer
        1st Defender's decision
    a1: Integer
       1st Attacker decision
    params: dict
        dict with the parameters for d1, a1, 
        

    Returns
        Float: Sample of theta1

    """

    p1, p0 = params[d1][a1]['p1'], params[d1][a1]['p0']

    sample = np.random.choice(a = [0,1],p = [p0,p1])
                                          
    return sample

def d1_sample(params):    ##### CHECK

    """
    Sample from D1

    Paramaters
    ----------
    params: dict
         parameters  (PROB FOR MULTINOMIAL)
        

    Returns
        Float: Sample of d1

    """

    
    return np.random.choice([1,2,3], p = params)

def theta2_sample(d2, theta1, a2, params):     #### CHECK


    p1, p0 = params[theta1][d2][a2]['p1'], params[theta1][d2][a2]['p0']

    sample = np.random.choice(a = [0,1],p = [p0,p1])

    return sample



def d2_sample(d1,a1,theta1, params):   #### CHECK
    """
    Sample from D2 regarding d1,a1, theta1, theta using params

    Paramaters
    ----------
    d1: Integer
        1st Defender's decision
    a1: Integer
       1st Attacker decision
    theta1: Integer
        Variable indicating if attack is succeful to anti aircraft missiles
    params: dict
        dict with the parameters for d1, a1, theta for probability
        

    Returns
        Float: Utility value of a1, theta and d2

    """

    p = params[theta1][d1][a1]

    return np.random.choice([1,2,3], p = p)



def propose(x_given, x_values):  #### CHECK

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


def inner_APS_A2(d1_given,theta1_given,a1_given, a2_values, a_util,d2,theta2, N_inner = 1000, burning = 0.75):  #### CHECK


    ### Inner APS for computing A2*(d1,theta1,a1)
    ### Params: 
    ### d1_given->Defender's decision d1
    ### theta1_given -> Defender's decision theta1
    ### a1_given-> Attacker's decision A1
    ### a2_values-> array-like, all possible values for A2
    ### a_util: callable, Attacker's utlity function
    ### d2: callable; fucntion to sample from pA(d2|d1,a1,theta1)
    ### theta2: function to sample from theta_2
    ### N_inner: number of iterations
    ### burnin: % of samples to discard


    a2_sim = np.zeros(N_inner, dtype = int)
    a2_sim[0] = np.random.choice(a2_values)

    d2_sim = d2(d1 = d1_given, theta1 =theta1_given , a1 = a1_given)
    theta2_sim = theta2(d2 = d2_sim, theta1 = theta1_given, a2 = a2_sim[0])

    for i in range(1, N_inner):

        ### proporse new attack a2
        a2_tilde = propose(a2_sim[i-1],a2_values)
        ### draw new sample from d2
        d2_tilde = d2(d1 = d1_given, theta1 =theta1_given , a1 = a1_given)
        ### draw new sample from theta2
        theta2_tilde = theta2(d2 = d2_tilde, theta1 = theta1_given, a2 = a2_tilde)

        num = a_util(d1 = d1_given, 
                    d2 = d2_tilde, 
                    theta1 = theta1_given, 
                    theta2 = theta2_tilde, 
                    a1 = a1_given,
                    a2 = a2_tilde)
        den = a_util(d1 = d1_given, 
                    d2 = d2_sim, 
                    theta1 = theta1_given, 
                    theta2 = theta2_sim, 
                    a1 =  a1_given,
                    a2 = a2_sim[i-1])


        if np.random.uniform() <= num/den:
            a2_sim[i] = a2_tilde
            theta2_sim = theta2_tilde
            d2_sim = d2_tilde

        else:
            a2_sim[i] = a2_sim[i-1]


    a2_dist = a2_sim[int(burning*N_inner):]

    return mode(a2_dist)[0], a2_dist


def compute_APSA2_i(d1_given,theta1_given,a1_given, a2_values,      
                    UA_params,D2_params,Theta2_params, idx,
                      N_inner_=1000, burnin_=0.75):     ### CHECK

    ## Compute one iteration of INNER APS of the attacker A2*(d1,theta1,a1)
    ## Function to wrap with joblib
    ## Params:
    ### d1_given->Defender's decision d1
    ### theta1_given -> Defender's decision theta1
    ### a1_given-> Attacker's decision A1
    ### a2_values-> array-like, all possible values for A2
    ### UA_params: array-like; params for random utilities UA
    ### D2_params: array-like; params for random utilities D2
    ### Theta2_params: array-like; params for random utilities Theta2
    ### idx: int; index to use for the array
    ### N_inner: number of iterations
    ### burning:% of samples to discard

    ua_i = partial(ua_sample,params = UA_params[idx])
    d2_i  = partial(d2_sample, params = D2_params[idx])
    theta2_i = partial(theta2_sample,params = Theta2_params[idx])


    return inner_APS_A2(d1_given = d1_given ,
                        theta1_given = theta1_given,
                        a1_given = a1_given, 
                        a2_values = a2_values, 
                        a_util = ua_i,
                        d2 = d2_i,
                        theta2 = theta2_i, 
                        N_inner = N_inner_, 
                        burning = burnin_)

def APS_A2_parallel(iters, d1_given,theta1_given,a1_given, a2_values, 
                    N_inner_=1000, burnin_=0.75, n_jobs = 2):                   #### CHECK
    

    ### Compute A*(d)- parallel style for one Defender's decision
    ### Params
    ### iters: int; samples for outer loop
    ### d1_given: int; d1 to compute 
    ### theta1_given: int, theta1 to compute
    ### a1_given: int, a1 to compute
    ### a2_values: array-like; Attacker's posiible decsions in A2
    ### N_inner: int; number of inner iterations
    ### burnin: float; % of samples to discard
    ### n_jobs: int; number of cores to use in parallel

    ua_params = UA_sample(iters)
    theta2_params = pAtheta2_sample(iters)
    d2_params = pAd2_sample(iters)

    r = Parallel(n_jobs = n_jobs)(delayed(compute_APSA2_i)(d1_given = d1_given,
                                                           theta1_given = theta1_given,
                                                           a1_given = a1_given, 
                                                           a2_values = a2_values, 
                                                       UA_params = ua_params,
                                                       D2_params = d2_params,
                                                       Theta2_params = theta2_params, 
                                                       idx = i,
                                                    N_inner_=N_inner_, 
                                                    burnin_=burnin_) for i in range(iters))
    

    mode_list, a2_dist_list = zip(*r)


    return np.concatenate(mode_list), a2_dist_list



def compute_A2_theta1_d1_a1(iters,                       
                            d1_values,
                            theta1_values,
                            a1_values,
                            a2_values,
                            N_inner_=1000, 
                            burnin_=0.75, 
                            n_jobs = 2):            #### CHECK
    ### Compute A2*(d1, theta1, a1) - parallel style
    ### Params
    ### iters int; samples from outer loop
    ### d1_values: array-like; Defender all possible decisions D1
    ### theta1_values: array-like; Theta1 support
    ### a1_values:array-like; A1 decisions
    ### a2_values: array-like; support of A2
    ### N_inner: int; number of inner iterations
    ### float; % of samples to discard
    ### n_jobs: int; number of cores to use in parallel

    A_theta1_d1_a1= {}

    for theta1 in theta1_values:
        theta1_aux = {}
        for d1 in d1_values:
            d1_aux = {}
            for a1 in a1_values:
                if (a1 ==1) & (theta1 ==1):
                    d1_aux[a1] = {'p':[-1,-1,-1], 'dist':[-1]}
                else:
                    mode_list, a_dist = APS_A2_parallel(iters = iters, 
                                    d1_given = d1,
                                    theta1_given = theta1,
                                    a1_given = a1, 
                                    a2_values = a2_values, 
                                    N_inner_= N_inner_, 
                                    burnin_= burnin_, 
                                    n_jobs = n_jobs)
                    
                    p = (np.bincount(np.array(mode_list).astype('int'), minlength=len(a2_values)+1) /iters)[1:]

                    d1_aux[a1] = {'p':list(p), 'dist':list(mode_list)}
            theta1_aux[d1] = d1_aux    
        A_theta1_d1_a1[theta1] = theta1_aux
    
    return A_theta1_d1_a1 



def inner_APS_D2_theta1_d1_a1(d2_values, 
                              a2_values,
                              theta1_given,
                                d1_given, 
                                a1_given,
                                d_util,
                                A2_theta1_d1_a1,
                                theta2,
                                N =1000,
                                burning = 0.75):               ### CHECK                           

    ### Compute d2*(theta1,d1,a1)
    ### Params:
    ### d2_values: array-likes: possible decisions of Defender at D2
    ### a2_values: array-likes: possible decisions of Attacker at A2
    ### theta1_given: integer; theta1 to compute
    ### d1_given: integer; d1 to computer
    ### a1_given: integer; a1 to compute
    ### d_util: callable; Defender's utility
    ### A2_theta1_d1_a1: dict: A2*(theta1,d1,a1)
    ### theta2: callable; pD(theta2|d2,theta1,a2)
    ### N: integer; n of samples
    ### burning: float; pct to samples to burn

    d2_sim = np.zeros(N, dtype = int)
    d2_sim[0] = np.random.choice(d2_values)
    a2_sim = np.random.choice(a2_values, 
                              p = A2_theta1_d1_a1[theta1_given][d1_given][a1_given]['p'])
    theta2_sim = theta2(d2 = d2_sim[0], a2 = a2_sim, theta1 = theta1_given)

    for i in range(1, N):
        #### propose new defense d2
        d2_tilde = propose(d2_sim[i-1], d2_values)
        #### draw new sample from attack a2
        a2_tilde = np.random.choice(a = np.array(a2_values), p = A2_theta1_d1_a1[theta1_given][d1_given][a1_given]['p'])
        #### draw new sample from theta2
        theta2_tilde = theta2(d2 = d2_tilde,
                              a2 = a2_tilde,
                              theta1 = theta1_given)
        
        ### input of D's utility: d1, theta, d2
        num = d_util(d1 = d1_given, 
                     d2 = d2_tilde,
                     theta1 = theta1_given,
                     theta2 = theta2_tilde,
                     a1 = a1_given,
                     a2 = a2_tilde)
        den = d_util(d1 = d1_given,
                     d2 = d2_sim[i-1],
                     theta1 = theta1_given,
                     theta2 = theta2_sim,
                     a1 = a1_given,
                     a2 = a2_sim)
        
        if np.random.uniform() <=num/den:
            d2_sim[i] = d2_tilde
            a2_sim = a2_tilde
            theta2_sim = theta2_tilde
        else:
            d2_sim[i] = d2_sim[i-1]

    d2_dist = d2_sim[int(burning * N):] 


    return {'mode':[(mode(d2_dist)[0])[0]], 'dist':list(d2_dist)}


def compute_APS_D2_theta1_d1_a1_parallel(d2_values,     
                              a2_values,
                              theta1_support,
                                d1_support, 
                                a1_support,
                                d_util,
                                A2_theta1_d1_a1,
                                theta2,
                                N =1000,
                                burning = 0.75,
                                n_jobs_ = 2):    #### CHECK
    

    ### Compute d2*(theta1,d1,a1) in parallel way
    ### Params:
    ### d2_values: array-likes: possible decisions of Defender at D2
    ### a2_values: array-likes: possible decisions of Attacker at A2
    ### theta1_support: integer; support of Theta1
    ### d1_support: integer; support of D1
    ### a1_support: integer; support of A1
    ### d_util: callable; Defender's utility
    ### A2_theta1_d1_a1: dict: A2*(theta1,d1,a1)
    ### theta2: callable; pD(theta2|d2,theta1,a2)
    ### N: integer; n of samples
    ### burning: float; pct to samples to burn
    ### n_jobs: number of processors

    comb = list(itertools.product(theta1_support, 
                                      d1_support, 
                                      a1_support))
    
    imp_cases = [i for i in comb if (i[0]==1 and i[2]==1)]
    possible_cases =  list(set(comb) - set(imp_cases))  
    
    r = Parallel(n_jobs = n_jobs_)(delayed(inner_APS_D2_theta1_d1_a1)(d2_values=d2_values, 
                              a2_values = a2_values,
                              theta1_given = theta1,
                                d1_given = d1, 
                                a1_given = a1,
                                d_util = d_util,
                                A2_theta1_d1_a1 = A2_theta1_d1_a1,
                                theta2 = theta2,
                                N = N,
                                burning = burning) for theta1, d1, a1 in possible_cases)

    ###### MAKE REORDER WITH COMB!!!!!!

    emp_d = {theta:{d:{a:{'d':[], 'dist':[]} for a in a1_support} for d in d1_support} for theta in theta1_support}

    for idx, x in enumerate(possible_cases):
        t1 = x[0]
        d1 = x[1]
        a1 = x[2]
        emp_d[t1][d1][a1]['d'] = r[idx]['mode'][0]
        emp_d[t1][d1][a1]['dist'] = r[idx]['dist']
    
    for idx, x in enumerate(imp_cases):
        t1 = x[0]
        d1 = x[1]
        a1 = x[2]
        emp_d[t1][d1][a1]['d'] = -1
        emp_d[t1][d1][a1]['dist'] = [-1]

    return emp_d


def inner_APS_A1(a1_values,a2_values ,a_util, d1, theta1, theta2, d2, A2_theta1_d1_a1,
                 N_inner = 1000, burnin = 0.75):       #### CHECK
    
    ### Inner APS for computing A2*
    ### Params: 
    ### a1_values -> Support of A1
    ### a2_values -> Support of A2
    ### a_util: callable; Attacker's utility function
    ### d1_util: callable; pA(d1) 
    ### theta_1: callable; pA(theta1|d1, a1)
    ### theta_2: callable; pA(theta2|a2,d2,theta2)
    ### d2: callable; pA(d2|d1,theta1,a1)
    ### A2_theta1_d1_a1: callable, generator optimal A2
    ### N_inner: number of iterations
    ### burning: % of samples to discard

    a1_sim = np.zeros(N_inner, dtype = int)
    a1_sim[0] = np.random.choice(a1_values)


    d1_sim = d1()
    theta1_sim = theta1(d1 = d1_sim, a1 = a1_sim[0])
    d2_sim = d2(d1 = d1_sim, a1 = a1_sim[0], theta1 = theta1_sim)
    a2_sim = np.random.choice(a2_values, 
                              p = A2_theta1_d1_a1[theta1_sim][d1_sim][a1_sim[0]]['p'])
    
    theta2_sim = theta2(d2 = d2_sim, a2 = a2_sim, theta1  = theta1_sim)

    for i in range(1, N_inner):

        ### propose new attack a1
        a1_tilde = propose(a1_sim[i-1],a1_values)
        ### draw new sample d1
        d1_tilde = d1()
        ### draw new sample from theta1
        theta1_tilde = theta1(d1 = d1_tilde, a1 = a1_tilde)
        ### draw new sample from d2_sim
        d2_tilde = d2(d1 = d1_tilde, a1 = a1_tilde, theta1 = theta1_tilde)
        ### draw new sample a2
        a2_tilde = np.random.choice(a2_values, 
                              p = A2_theta1_d1_a1[theta1_tilde][d1_tilde][a1_tilde]['p'])
    
        ### draw new sample theta2
        theta2_tilde = theta2(d2 = d2_tilde, 
                              a2 = a2_tilde, 
                              theta1 = theta1_tilde)
        
        num = a_util(d1 = d1_tilde,
                     d2 = d2_tilde,
                     theta1 = theta1_tilde,
                     theta2  = theta2_tilde,
                     a1 = a1_tilde,
                     a2 = a2_tilde)
        
        den = a_util(d1 = d1_sim,
                     d2 = d2_sim,
                     theta1  = theta1_sim,
                     theta2 = theta2_sim,
                     a1 = a1_sim[i-1], 
                     a2 = a2_sim)
        
        if np.random.uniform() <= num/den:
            a1_sim[i] = a1_tilde
            theta1_sim  = theta1_tilde
            d1_sim = d1_tilde
            a2_sim = a2_tilde
            d2_sim = d2_tilde
            theta2_sim = theta2_tilde
        else:
            a1_sim[i] = a1_sim[i-1]
    
    a1_dist = a1_sim[int(burnin*N_inner):]
    return mode(a1_dist)[0], a1_dist


def compute_APS_A1_i(a1_values,
                     a2_values,
                     ua_params,
                     d1_params,
                     theta1_params,
                     theta2_params,
                     d2_params, 
                     A2_theta1_d1_a1,
                     idx,
                     N_inner = 1000, 
                     burnin = 0.75):          #### CHECK

    ## Compute one iteration of INNER APS A1 of the attacker A1*
    ### Function to wrap with joblib
    ### Params:
    ### a1_values -> Support of A1
    ### a2_values -> Support of A2
    ### ua_params: array-like; params for random utilities UA
    ### d1_params: array-like; params for random prob pA(d1) 
    ### theta1_params: array-like; params for random prob pA(theta1|d1, a1)
    ### theta2_params: array-like; params for random prob pA(theta2|a2,d2,theta2)
    ### d2_params: array-like; params for random prob pA(d2|d1,theta1,a1)
    ### A2_theta1_d1_a1: callable, generator optimal A2
    ### idx: int; index to use for the array
    ### N_inner: number of iterations
    ### burning: % of samples to discard

    d1_i  = partial(d1_sample, params = d1_params[idx])
    theta1_i = partial(theta1_sample, params = theta1_params[idx])
    d2_i = partial(d2_sample, params = d2_params[idx])
    theta2_i = partial(theta2_sample, params = theta2_params[idx])
    ua_i = partial(ua_sample, params =  ua_params[idx])


    return inner_APS_A1(a1_values = a1_values,
                 a2_values = a2_values,
                 a_util = ua_i,
                 d1 = d1_i,
                 theta1 = theta1_i, 
                 theta2 = theta2_i, 
                 d2 = d2_i, 
                 A2_theta1_d1_a1 = A2_theta1_d1_a1,
                 N_inner = N_inner, 
                 burnin = burnin)



def APS_A1_parallel(iters,a1_values,a2_values, A2_theta1_d1_a1, N_inner = 1000, burning = 0.75, n_jobs = 2):   #### CHECK

    ### Compute A*- parallel style
    ### Params
    ### iters: int; samples for outer loop
    ### a1_values: array-like; Attacker's posiible decsions in A1
    ### a2_values: array-like; Attacker's posiible decsions in A2
    ### N_inner: int; number of inner iterations
    ### burnin: float; % of samples to discard
    ### n_jobs: int; number of cores to use in paller

    d1_params = pAd1_sample(iters)
    theta1_params= pAtheta1_sample(iters)
    d2_params = pAd2_sample(iters)
    theta2_params = pAtheta2_sample(iters)
    ua_params = UA_sample(iters)

    r = Parallel(n_jobs = n_jobs)(delayed(compute_APS_A1_i)(a1_values = a1_values,
                    a2_values = a2_values,
                     ua_params = ua_params,
                     d1_params = d1_params,
                     theta1_params = theta1_params,
                     theta2_params = theta2_params,
                     d2_params  = d2_params, 
                     A2_theta1_d1_a1 = A2_theta1_d1_a1,
                     idx = i,
                     N_inner = N_inner, 
                     burnin = burning) for i in range(iters))
    
    mode_list, a_dist_list = zip(*r)

    return np.concatenate(mode_list), a_dist_list




def compute_A1(iters, a1_values, a2_values, A2_theta1_d1_a1, N_inner_=1000, burnin_=0.75, n_jobs = 2):     #### CHCEKC
 
    ### Compute A1* 
    ### Params
    ### iters: int; samples for outer loop
    ### a1_values: array-like; Defender's possible decisions A1
    ### a2_values: array-like; Attacker's posiible decsions A2
    ### A2_theta1_d1_a1: dict; A*2(theta1,d1,a1)
    ### N_inner: int; number of inner iterations
    ### burnin: float; % of samples to discard
    ### n_jobs: int; number of cores to use in parallel

    mode_list,a_dist = APS_A1_parallel(iters = iters,
                    a1_values = a1_values,
                    a2_values = a2_values, 
                    A2_theta1_d1_a1 = A2_theta1_d1_a1, 
                    N_inner = N_inner_, 
                    burning = burnin_, 
                    n_jobs = n_jobs)
    
    p = (np.bincount(np.array(mode_list).astype('int'), minlength=len(a1_values)+1) /iters)[1:]


    return  {'p':list(p), 'dist':list(mode_list)}


def inner_APS_D1(d1_values,a1_values,a2_values, A1, theta1,theta2,d2_opt, A2_theta1_d1_a1, d_util, N = 1000, burnin = 0.75):  ##CHECK

    ### Compute d1* opt
    ### Params:
    ### d1_values:  array-likes: possible decsions of Defender at D1
    ### a1_values:  array-likes: possible decisions of Attacker at D1
    ### a2_values:  array-likes: possible decisions of Attacker at A2
    ### A1: dict, dict with the probs to sample pD(a1)
    ### theta1: callable, pD(theta1|d1,a1)
    ### theta2: callable, pD(theta2|d2,a2,theta1)
    ### d2_opt: callable, d2(d1,theta1,a1)
    ### A2_theta1_d1_a1: dict, A2*(d1,theta1,a1)
    ### d_util: callable; Defender's utility
    ### N: integer; n of samples
    ### burning: float; pct to saaples to burn



    d1_sim = np.zeros(N, dtype = int)
    d1_sim[0] = np.random.choice(d1_values)
    a1_sim = np.random.choice(a1_values, p = A1['p'])
    theta1_sim = theta1(d1 = d1_sim[0], a1 = a1_sim)
    d2_sim = d2_opt[theta1_sim][d1_sim[0]][a1_sim]['d']
    a2_sim = np.random.choice(a2_values, 
                              p = A2_theta1_d1_a1[theta1_sim][d1_sim[0]][a1_sim]['p'])
    theta2_sim = theta2(d2 = d2_sim, 
                        a2 = a2_sim, 
                        theta1  = theta1_sim)
    
    for i in range(1,N):

        ### propose new defense d1
        d1_tilde = propose(d1_sim[i-1],d1_values)
        ### draw new sample from attack a1
        a1_tilde = np.random.choice(a1_values, p = A1['p'])
        ### draw new sample from theta1
        theta1_tilde = theta1(d1 = d1_tilde, a1 = a1_tilde)
        ### draw new sample from d2
        d2_tilde = d2_opt[theta1_tilde][d1_tilde][a1_tilde]['d']
        ### draw new sample from a2
        a2_tilde = np.random.choice(a2_values, 
                              p = A2_theta1_d1_a1[theta1_tilde][d1_tilde][a1_tilde]['p'])
        
        ### draw new sample from theta2
        theta2_tilde = theta2(d2 = d2_tilde, 
                        a2 = a2_tilde, 
                        theta1  = theta1_tilde)
        
        num = d_util(d1 = d1_tilde,
                     d2 = d2_tilde,
                     theta1 = theta1_tilde,
                     theta2 = theta2_tilde,
                     a1 = a1_tilde,
                     a2 = a2_tilde)
        
        den = d_util(d1 = d1_sim[i-1],
                     d2 = d2_sim,
                     theta1 = theta1_sim,
                     theta2 = theta2_sim,
                     a1 = a1_sim,
                     a2 = a2_sim)
        
        if np.random.uniform() <= num/den:
            d1_sim[i] = d1_tilde
            a1_sim = a1_tilde
            theta1_sim = theta1_tilde
            d2_sim = d2_tilde
            a2_sim = a2_tilde
            theta2_sim = theta2_tilde
        else:
            d1_sim[i] = d1_sim[i-1]

    d1_dist = d1_sim[int(burnin*N):]

    return {'mode':[(mode(d1_dist)[0])[0]], 'dist':list(d1_dist)}







    
    


