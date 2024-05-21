import pandas as pd
import numpy as np


def cD(d1, theta, d2, a1):
  """
  Computes the monetary value depending on d1, d2 and theta

  Paramaters
  ----------
  d1: Integer
    1st Defender's decision
  theta: Integer
    Variable indicating if attack is succeful
  d2: Integer
    2nd Defender's decision
  a1: Integer
    1st Attacker decision
   Returns
    Float: Monetary value of consequnces of d1, theta and d2

  """
  cD_arr = [{'d1':1,'theta':1,'d2':1,'cD':15.16},
  {'d1': 1,'theta':1,'d2':2,'cD':2.3},
  {'d1':1,'theta':1,'d2':3,'cD':4.28},
  {'d1':1,'theta': 0,'d2':'NA','cD':0},
  {'d1': 2,'theta':1,'d2':1,'cD':17.25},
  {'d1':2,'theta':1,'d2':2,'cD':4.39},
  {'d1':2,'theta':1,'d2':3,'cD':6.37},
  {'d1':2,'theta':0,'d2':'NA','cD':0.05},
  {'d1':3,'theta':1,'d2':1,'cD':19.39},
  {'d1':3,'theta':1,'d2':2,'cD':6.53},
  {'d1':3,'theta':1,'d2':3,'cD':8.51},
  {'d1':3,'theta':0 ,'d2': 'NA','cD':0.15},
  {'d1':4,'theta': 'NA' ,'d2':'NA','cD':0.5}]

  cD_df = pd.DataFrame(cD_arr)
  if a1 !=1:
    theta = 0
  if d1 ==4:
    cD_value = cD_df[cD_df['d1']==d1]['cD'].values[0]
  else:
    if theta == 0:
      d2 = 'NA'
    cD_value = cD_df[(cD_df['d1']==d1)&(cD_df['theta']==theta)&(cD_df['d2']==d2)]['cD'].values[0]

  return cD_value



def cA(a1, theta, d2, d1):

    """
    Computes the monetary value depending on a1, theta and d2
    
    Paramaters
    ----------
    a1: Integer
        Attacker's decision
    theta: Integer
        Variable indicating if attack is succeful
    d2: Integer
        2nd Defender's decision
    d1: Integer
        1st Defender's decision

    Returns
        Float: Monetary value of consequnces of a1, theta and d2
    """
    if (a1 == 1) & (d1==4):
      cA_value = -10
    else:
      aD_arr = [{'a1': 0,'theta': 'NA','d2':'NA', 'cA': 0},
                {'a1': 'ATT','theta': 1,'d2':1, 'cA':0.97},
                {'a1': 'ATT','theta':1,'d2':2,'cA':2.27},
                {'a1': 'ATT','theta':1,'d2':3,'cA':-1.28},
                {'a1': 'ATT','theta': 0,'d2':'NA','cA':-0.53}]

      cA_df = pd.DataFrame(aD_arr)

      if a1 == 0:
        theta = 'NA'
        d2 = 'NA'
      else:
        a1 = 'ATT'
        if theta == 0:
          d2 = 'NA'

      cA_value = cA_df[(cA_df['a1']==a1)&(cA_df['theta']==theta)&(cA_df['d2']==d2)]['cA'].values[0]

    return cA_value


def pAtheta_sample(N):

    """
    Generates the params to generate N random probabily samples 
    PA(Theta|D1,A1). 
    
    Parameters
    ------------
    N: Integer

    Returns:

    Array-like: 
        List with N dictionarie containg the parameters for pA(theta|d1,a1)
    
    """

    params_arr = []
    ## Attack to another boat
    th_1_ai = np.random.beta(a =1 , b = 1, size = (N,3))
    th_0_ai = 1 - th_1_ai
    ### Attack to our boat with D1 = 1
    th_1_a1_d1_1 = np.random.beta(a =40, b = 60, size = N)
    th_0_a1_d1_1 = 1 - th_1_a1_d1_1
    ### Attack to our boat with D1 = 2
    th_1_a1_d1_2 = np.random.beta(a =10, b = 90, size = N)
    th_0_a1_d1_2 = 1 - th_1_a1_d1_2
    ### Attack to our boat with D1 = 3
    th_1_a1_d1_3 = np.random.beta(a = 50, b = 950, size = N)
    th_0_a1_d1_3 = 1 - th_1_a1_d1_3

    for n in range(N):
      params = {}
      for d1 in [1,2,3,4]:
        params_di = {}
        # no attack
        params_di[0] =  {'p0': 1, 'p1': 0}
        ## another boat
        p1_vec = th_1_ai[n]
        p0_vec = th_0_ai[n]
        for idx,a in enumerate([2,3,4]):
          p1, p0 = p1_vec[idx],p0_vec[idx]
          params_di[a] = {'p0': p0, 'p1': p1}
        ## our boat
        if d1 == 1:
          p1,p0 = th_1_a1_d1_1[n], th_0_a1_d1_1[n]
        elif d1 == 2:
          p1,p0 = th_1_a1_d1_2[n], th_0_a1_d1_2[n]
        elif d1 ==3:
          p1,p0 = th_1_a1_d1_3[n], th_0_a1_d1_3[n]
        elif d1 ==4:
          p1, p0 = 0, 1

        params_di[1] = {'p0': p0, 'p1': p1}
        params[d1] = params_di
      params_arr.append(params)

    return params_arr

def UA_sample(N):
  
    """
    Generates the params to generate N random utility samples 
    UA(cA). 
    
    Parameters
    ------------
    N: Integer

    Returns:

    Array-like: 
        List with N dictionarie containg the parameters for uA(cA)
    
    """

    return np.random.uniform(0,20, size=N)

def pAd2_sample(N):
  """
    Generates the params to generate N random probabily samples 
    PA(D2|D1,A,THETA). 
    
    Parameters
    ------------
    N: Integer

    Returns:

    Array-like: 
        List with N dictionaries containg the parameters for pA(d2|d1,a,theta)
    
    """

  params_arr = [] ## d,a,theta
  ai_th1 = np.random.dirichlet((1,1,1),size = 3 * N)
  d1_a1_th1 = np.random.dirichlet((1,1,1),size = N)
  d2_a1_th1 = np.random.dirichlet((0.1,4,6),size = N)
  d3_a1_th1 = np.random.dirichlet((0.1,1,10),size = N)
  for n in range(N):
    params = {}
    for d1 in [1,2,3,4]:
      params_di = {}
      # no attack
      params_di[0] = {0:[1,0,0],
                      1:[1,0,0]}
      ## attack to others
      for idx,ai in enumerate([2,3,4]):
        params_di[ai] = {0:[1,0,0],
                         1:ai_th1[idx+n]}

      ## attack to our boat
      if d1 == 1:
        params_di[1] = {0: [1,0,0],
                        1: d1_a1_th1[n]}
      elif d1 ==2:
          params_di[1] = {0: [1,0,0],
                        1: d2_a1_th1[n]}
      elif d1 ==3:
        params_di[1] = {0: [1,0,0],
                        1: d3_a1_th1[n]}
      elif d1 ==4:
        params_di[1] = {0:[1,0,0],
                      1:[1,0,0]}

      params[d1] = params_di
    params_arr.append(params)

  return params_arr

def pDtheta():

    """
    Generates the params to generate N random probabily samples 
    pD(theta|d1,a1). 
    
    Parameters
    ------------
    N: Integer

    Returns:

    Array-like: 
        List with N dictionarie containg the parameters for pD(theta|d1,a1)
    
    """

    params = {}
    for d1 in [1,2,3,4]:
        params_di = {}
        # no attack
        params_di[0] =  {'p0': 1, 'p1': 0}
        ## another boat
        for idx,a in enumerate([2,3,4]):
          params_di[a] = {'p0': 0.5, 'p1': 0.5}
        ## our boat
        if d1 == 1:
          p1,p0 = 0.4,0.6
        elif d1 == 2:
          p1,p0 = 0.1,0.9
        elif d1 ==3:
          p1,p0 = 0.05,0.95
        elif d1 ==4:
          p1, p0 = 0, 1

        params_di[1] = {'p0': p0, 'p1': p1}
        params[d1] = params_di

    return params