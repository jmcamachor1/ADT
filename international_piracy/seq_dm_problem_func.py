import numpy as np
import pandas as pd
from functools import partial
from scipy.stats import mode
from joblib import Parallel, delayed
from param_func import cD, cA, pDtheta ,pAtheta_sample, UA_sample, pAd2_sample


def uD(d1, theta, d2,  c):
    """
    Computes the Defender's utility depending on d1, d2 and theta

    Paramaters
    ----------
    d1: Integer
        1st Defender's decision
    theta: Integer
        Variable indicating if attack is succeful
    d2: Integer
        2nd Defender's decision
    c: Float
        Risk averse coefficient

        Returns
        Float: Utility value of d1, theta and d2

    """

    alpha_D = np.exp(c* 20) ## remain utility positive

    return - np.exp([c * cD(d1, theta, d2)]) + alpha_D


def ua_sample(a1,theta,d2,d1,params):
  
  """
    Computes the Attakcr's utility depending on a1, theta and d2

    Paramaters
    ----------
    a1: Integer
        Attcker's decision
    theta: Integer
        Variable indicating if attack is succeful
    d2: Integer
        2nd Defender's decision
    params: Float
        Risk averse coefficient

        Returns
        Float: Utility value of a1, theta and d2

  """
    
  return np.exp([params * cA(a1, theta, d2,d1)])



def theta_sample(d,a, params):

    """
    Computes the Attakcr's utility depending on a1, theta and d2

    Paramaters
    ----------
    a1: Integer
        Attcker's decision
    theta: Integer
        Variable indicating if attack is succeful
    d2: Integer
        2nd Defender's decision
    params: Float
        Risk averse coefficient

        Returns
        Float: Utility value of a1, theta and d2

    """
    p0,p1 = params[d][a]['p0'], params[d][a]['p1']
    sample = np.random.choice(a = [0,1],p = [p0,p1])
       
    return sample


def d2_sample(d,a,theta, params):
    """
    Sample from D2 regarding a, theta using params

    Paramaters
    ----------
    d: Integer
        1st Defender's decision
    a: Integer
        Attacker decision
    theta: Integer
        Variable indicating if attack is succeful
    params: Float
        Risk averse coefficient

        Returns
        Float: Utility value of a1, theta and d2

    """

    p = params[d][a][theta]

    return np.random.choice([1,2,3], p = p)

def propose(x_given, x_values):

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


def generate_df_uD(c):
  
  ## Generate dataframe with all the possible utilities of the Defender

  l = []
  d1_l = [1,2,3,4]
  d2_l = [1,2,3]
  theta_l = [0,1]

  for d1 in d1_l:
    for d2 in d2_l:
      for theta in theta_l:
        l.append({'d1':d1, 'd2':d2, 'theta':theta, 'uD': uD(d1 = d1,
                                                   theta = theta,
                                                   d2= d2,
                                                   c = c)[0]})
  return pd.DataFrame(l)


def APS_D2(d1, theta, c):

    ### COMPUTE OPTIMAL DECISION D2 given D1 and THETA; 
    ## FOR OUR INTERNATIONAL PIRACY PROBLEM IS DETERMINISTIC
    ### PARAMS: d1 -> Integer theta-> integer c-> float

    df = generate_df_uD(c)
    argmax_d2 = df[(df['d1']==d1) & (df['theta']==theta)].sort_values(by='uD', ascending=False).iloc[0]['d2']
    return argmax_d2


def inner_APS_A(d_given, a_values, a_util, theta, d2,  N_inner=1000, burnin=0.75):

    ### Inner APS for computing A*(d)
    ### Params: d_given -> Defender's decision to compute A*(d)
    ### a_values: array-like; all possible Attacker decisions
    ### a_util: callable; Attacker's utility function
    ### theta: callable; pA(theta|d,a )
    ### d2: callable; pA(d2|d,a,theta)
    ### N_inner: number of iterations
    ### burnin: % of samples to discard

    a_sim = np.zeros(N_inner, dtype = int)
    a_sim[0] = np.random.choice(a_values)
    
    theta_sim = theta(d_given, a_sim[0])
    d2_sim = d2(d_given, a_sim[0], theta_sim)

    for i in range(1, N_inner):
      ## propose new attack a
      a_tilde = propose(a_sim[i-1],a_values)
      ## draw new sample from theta
      theta_tilde = theta(d_given, a_tilde)
      ## draw new sample from d2
      d2_tilde = d2(d_given, a_tilde, theta_tilde)
      num = a_util(a_tilde, theta_tilde, d2_tilde, d_given)
      den = a_util(a_sim[i-1], theta_sim, d2_sim, d_given)

      if np.random.uniform() <= num/den:
        a_sim[i] = a_tilde
        theta_sim = theta_tilde
        d2_sim = d2_tilde
      else:
        a_sim[i] = a_sim[i-1]

    a_dist = a_sim[int(burnin*N_inner):]
    return mode(a_dist)[0], a_dist


def compute_APSA_i(d_given, a_values, UA_params,theta_params,d2_params, idx, N_inner_=1000, burnin_=0.75):
  
  ## Compute one iteration of INNER APS of the attacker A*(d)
  ### Function to wrap with joblib
  ### Params:
  ###d_given -> Defender's decision to compute A*(d)
  ### a_values: array-like; all possible Attacker decisions
  ### UA_params: array-like; params for random utilities UA
  ### theta_params: array-like; params for random prob pA(theta|d,a)
  ### d2_params: array-like; params for pA(D2|d,a,theta)
  ### idx: int; index to use for the array
  ### N_inner: number of iterations
  ### burnin: % of samples to discard

  theta_i = partial(theta_sample,params = theta_params[idx])
  ua_i = partial(ua_sample,params = UA_params[idx])
  d2_i  = partial(d2_sample, params = d2_params[idx])
  return inner_APS_A(d_given = d_given, a_values = a_values, a_util = ua_i, theta = theta_i, d2 = d2_i,  N_inner=N_inner_, burnin=burnin_)

def APS_A_parallel(iters,d_given, a_values,N_inner_=1000, burnin_=0.75, n_jobs = 2):
  
  ### Compute A*(d)- parallel style for one Defender's decision
  ### Params
  ### iters: int; samples for outer loop
  ### d_given: int; d to compute 
  ### a_values: array-like; Attacker's posiible decsions 
  ### N_inner: int; number of inner iterations
  ### burnin: float; % of samples to discard
  ### n_jobs: int; number of cores to use in paller


  UA_params = UA_sample(iters)
  theta_params = pAtheta_sample(iters)
  d2_params = pAd2_sample(iters)
  
  r = Parallel(n_jobs = n_jobs)(delayed(compute_APSA_i)(d_given = d_given, 
                 a_values = a_values, 
                 UA_params = UA_params,
                 theta_params = theta_params,
                 d2_params = d2_params, 
                 idx = i, 
                 N_inner_=N_inner_, 
                 burnin_=burnin_) for i in range(iters))
  mode_list, a_dist_list = zip(*r)
  
  return np.concatenate(mode_list), a_dist_list



def compute_A_d(iters,d_values, a_values,N_inner_=1000, burnin_=0.75, n_jobs = 2):
  
  ### Compute A*(d)- parallel style for all decision
  ### Params
  ### iters: int; samples for outer loop
  ### d_values: array-like; Defender's possible decisions 
  ### a_values: array-like; Attacker's posiible decsions 
  ### N_inner: int; number of inner iterations
  ### burnin: float; % of samples to discard
  ### n_jobs: int; number of cores to use in parallel

  A_d = {}
  for d in d_values:
    mode_list,a_dist =  APS_A_parallel(iters = iters,
                    d_given = d, 
                    a_values = a_values,
                    N_inner_=N_inner_, 
                    burnin_=burnin_, n_jobs = n_jobs)
    
    p = np.bincount(np.array(mode_list).astype('int'), minlength=len(a_values)) /iters
    A_d[d] = {'p':list(p), 'dist':list(mode_list)}

  return A_d



def inner_APS_D1(d_values, a_values, d_util ,A_d, theta, d2_opt, N = 1000, burnin = 0.75):
  
  ### Compute d1* opt
  ### Params:
  ### d_values:  array-likes: possible decsions of Defender at D1
  ### a_values: array-like: possible decisions of Attacker
  ### d_util: callable; Defender's utility
  ### A_d: dict: A*(d)
  ### theta; callable: pD(theta|d,a)
  ### d2_opt: callable; D2*(d1, theta)
  ### N: integer; n of samples
  ### burning: float; pct to smaples to burn

  d1_sim = np.zeros(N, dtype = int)
  d1_sim[0] = np.random.choice(d_values)
  a_sim = np.random.choice(a_values, p = A_d[d1_sim[0]]['p'])
  theta_sim = theta(d1_sim[0],a_sim)
  d2_sim = d2_opt(d1 = d1_sim[0], theta = theta_sim)

  for i in range(1,N):

    ### propose new defense
    d_tilde = propose(d1_sim[i-1], d_values)
    ### draw new sample from attack
    a_tilde = np.random.choice(a_values, p = A_d[d_tilde]['p'])
    ### draw new sample from theta
    theta_tilde = theta(d_tilde, a_tilde)
    ## draw optimal d2
    d2_tilde = d2_opt(d1 = d_tilde, theta = theta_tilde)

    #input of D's utility: d1, theta, d2
    num = d_util(d_tilde , theta_tilde , d2_tilde)
    den = d_util(d1_sim[i-1], theta_sim, d2_sim)

    if np.random.uniform() <= num/den:
        d1_sim[i] = d_tilde
        theta_sim = theta_tilde
        d2_sim = d2_tilde
    else:
      d1_sim[i] = d1_sim[i-1]

  d_dist = d1_sim[int(burnin*N):]

  return {'mode':[(mode(d_dist)[0])[0]], 'dist':list(d_dist)}

