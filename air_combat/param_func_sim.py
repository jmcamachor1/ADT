import numpy as np


def cD(d1,d2,theta1,theta2,a1,a2):    ##### CHECK
        """
        Computes the monetary value depending on d1, d2 and theta

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


        Returns
        Float: Monetary value of consequnces of d1,d2, theta1,theta2 and a1,a2

        """
   
        fD_d1 = {1:0,
            2:0.3,
            3: 0.6}

        fD_a1 = {1:0,
            2:0.4,
            3:0.5}

        fD_theta1 = {0:0,1:2}

        fD_d2 = {1:0, 2:0.1,3:0.2}

        fD_a2 = {1:0, 2:0.5, 3:0.7}

        fD_theta_2 = {0:0, 1:8}

        cD_val = fD_d1[d1] + fD_a1[a1] + fD_theta1[theta1] +fD_d2[d2] + fD_a2[a2] + fD_theta_2[theta2]

        return cD_val


def cA(d1,d2,theta1,theta2,a1,a2):   ##### CHECK
        """
        Computes the monetary value depending on d1,d2, theta1,theta2 and a1,a2

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


        Returns
        Float: Monetary value of consequnces of d1,d2, theta1,theta2 and a1,a2

        """

        fA_d1 = {1:0,2:0.4,3:0.8}

        fA_a1 = {1:0,2:0.2,3:0.3}

        fA_theta1 = {0:0,1:1.5}

        fA_d2 = {1:0, 2:0.5, 3:0.8}

        fA_a2 = {1:0, 2:0.4, 3:0.6}

        fA_theta2 = {0:0, 1:4}


        cA_val = fA_d1[d1] + fA_a1[a1] + fA_theta1[theta1] +fA_d2[d2] + fA_a2[a2] + fA_theta2[theta2]

        return cA_val


def pAtheta1_sample(N):   #### CHECK

    """
    Generates the params to generate N random probabily samples 
    PA(Theta1|D1,A1). 
    
    Parameters
    ------------
    N: Integer

    Returns:

    Array-like: 
        List with N dictionarie containg the parameters for pA(theta1|d1,a1)
    
    """

    ### D1 = 1, A1 = 2
    th_1_d1_1_a1_2 = np.random.beta(a = 80, b = 20, size = N)
    th_0_d1_1_a1_2 = 1 - th_1_d1_1_a1_2
    ### D1 = 2, A1 = 2
    th_1_d1_2_a1_2 = np.random.beta(a = 70, b = 30, size = N)
    th_0_d1_2_a1_2 = 1 - th_1_d1_2_a1_2
    ### D1 = 3, A1 = 2
    th_1_d1_3_a1_2 = np.random.beta(a = 20, b = 80, size = N)
    th_0_d1_3_a1_2 = 1 - th_1_d1_3_a1_2
    ### D1 = 1, A1 = 3
    th_1_d1_1_a1_3 = np.random.beta(a = 999, b = 1, size = N)
    th_0_d1_1_a1_3 = 1 - th_1_d1_1_a1_3
    ### D1 = 2, A1 = 3
    th_1_d1_2_a1_3 = np.random.beta(a = 60, b = 40, size = N)
    th_0_d1_2_a1_3 = 1 - th_1_d1_2_a1_3
    ### D1 = 3, A1 = 3
    th_1_d1_3_a1_3 = np.random.beta(a =40 , b = 60 , size = N)
    th_0_d1_3_a1_3 = 1 - th_1_d1_3_a1_3

    params_arr = []
    for n in range(N):
        params = {}
        ### ---------------------NO DEFENSE-----------####
        params_d1 = {}
        #### NO ATTACK
        params_d1[1] = {'p0':1, 'p1':0}
        ### LOW ATTACK
        p1, p0 = th_1_d1_1_a1_2[n], th_0_d1_1_a1_2[n]
        params_d1[2] = {'p0':p0, 'p1':p1}
        ### HIGH ATTACK
        p1, p0 = th_1_d1_1_a1_3[n], th_0_d1_1_a1_3[n]
        params_d1[3] = {'p0':p0, 'p1':p1}
        params[1] = params_d1

        ### ---------------------MINIMAL DEFENSE--------###
        params_d2 = {}
        #### NO ATTACK
        params_d2[1] = {'p0':1, 'p1':0}
        ### LOW ATTACK
        p1, p0 = th_1_d1_2_a1_2[n], th_0_d1_2_a1_2[n]
        params_d2[2] = {'p0':p0, 'p1':p1}
        ### HIGH ATTACK
        p1, p0 = th_1_d1_2_a1_3[n], th_0_d1_2_a1_3[n]
        params_d2[3] = {'p0':p0, 'p1':p1}
        params[2] = params_d2


        ### ----------------------HIGH DEFENSE------------###
        params_d3 = {}
        #### NO ATTACK
        params_d3[1] = {'p0':1, 'p1':0}
        ### LOW ATTACK
        p1, p0 = th_1_d1_3_a1_2[n], th_0_d1_3_a1_2[n]
        params_d3[2] = {'p0':p0, 'p1':p1}
        ### HIGH ATTACK
        p1, p0 = th_1_d1_3_a1_3[n], th_0_d1_3_a1_3[n]
        params_d3[3] = {'p0':p0, 'p1':p1}
        params[3] = params_d3

        params_arr.append(params)

    return params_arr

def pDtheta1():    ## CHECK

    """
    Generates the params to generate a sample pD(theta1|d1,a1). 
    
    Parameters
    ------------

    Returns:

    Array-like: 
        dictionarY containg the parameters for pD(theta1|d1,a1)
    
    """
    ############################# NO DEFENSE ##########################
    param_d1 = {}
    # NO ATTACK 
    param_d1[1] = {'p0': 1, 'p1': 0}
    ## LOW ATTACK
    param_d1[2] = {'p0': 0.2, 'p1': 0.80}
    ## HIGH ATTACK
    param_d1[3] = {'p0': 0, 'p1': 1}
    ############################# LOW DEFENSE ##########################
    param_d2 = {}
    # NO ATTACK 
    param_d2[1] = {'p0': 1, 'p1': 0}
    ## LOW ATTACK
    param_d2[2] = {'p0': 0.70, 'p1': 0.30}
    ## HIGH ATTACK
    param_d2[3] = {'p0': 0.40, 'p1': 0.60}
    ############################# HIGH DEFENSE ##########################
    param_d3 = {}
    # NO ATTACK 
    param_d3[1] = {'p0': 1, 'p1': 0}
    ## LOW ATTACK
    param_d3[2] = {'p0':0.80 , 'p1': 0.20}
    ## HIGH ATTACK
    param_d3[3] = {'p0':0.60 , 'p1': 0.40}


    return {1:param_d1, 2:param_d2, 3:param_d3}



def pAtheta2_sample(N):  ### CHECK

    """
    Generates the params to generate N random probabily samples 
    PA(Theta2|D1,A1,theta1). 
    
    Parameters
    ------------
    N: Integer

    Returns:

    Array-like: 
        List with N dictionarie containg the parameters for pA(theta1|d1,a1)
    
    """


    ######------------THETA1 = 1-------------------------------####
    ### D2 = 1, A2 = 2
    th2_1_d2_1_a2_2_th1_1 = np.random.beta(a = 90, b = 10, size = N)
    th2_0_d2_1_a2_2_th1_1 = 1 - th2_1_d2_1_a2_2_th1_1
    ### D2 = 2, A2 = 2
    th2_1_d2_2_a2_2_th1_1 = np.random.beta(a = 70, b = 30, size = N)
    th2_0_d2_2_a2_2_th1_1 = 1 - th2_1_d2_2_a2_2_th1_1
    ### D2 = 3, A2 = 2
    th2_1_d2_3_a2_2_th1_1 = np.random.beta(a = 30, b = 70, size = N)
    th2_0_d2_3_a2_2_th1_1 = 1 - th2_1_d2_3_a2_2_th1_1
    ### D2 = 1, A2 = 3
    th2_1_d2_1_a2_3_th1_1 = np.random.beta(a = 999, b = 1, size = N)
    th2_0_d2_1_a2_3_th1_1 = 1 - th2_1_d2_1_a2_3_th1_1
    ### D2 = 2, A2 = 3
    th2_1_d2_2_a2_3_th1_1 = np.random.beta(a = 90, b = 10, size = N)
    th2_0_d2_2_a2_3_th1_1 = 1 - th2_1_d2_2_a2_3_th1_1
    ### D2 = 3, A2 = 3
    th2_1_d2_3_a2_3_th1_1 = np.random.beta(a = 60, b = 40, size = N)
    th2_0_d2_3_a2_3_th1_1 = 1 - th2_1_d2_3_a2_3_th1_1

    ######------------THETA1 = 0-------------------------------####
    ### D2 = 1, A2 = 2
    th2_1_d2_1_a2_2_th1_0 = np.random.beta(a = 45, b = 55, size = N)
    th2_0_d2_1_a2_2_th1_0 = 1 - th2_1_d2_1_a2_2_th1_0
    ### D2 = 2, A2 = 2
    th2_1_d2_2_a2_2_th1_0 = np.random.beta(a = 35, b = 65, size = N)
    th2_0_d2_2_a2_2_th1_0 = 1 - th2_1_d2_2_a2_2_th1_0
    ### D2 = 3, A2 = 2
    th2_1_d2_3_a2_2_th1_0 = np.random.beta(a = 15, b = 85, size = N)
    th2_0_d2_3_a2_2_th1_0 = 1 - th2_1_d2_3_a2_2_th1_0
    ### D2 = 1, A2 = 3
    th2_1_d2_1_a2_3_th1_0 = np.random.beta(a = 50, b = 50, size = N)
    th2_0_d2_1_a2_3_th1_0 = 1 - th2_1_d2_1_a2_3_th1_0
    ### D2 = 2, A2 = 3
    th2_1_d2_2_a2_3_th1_0 = np.random.beta(a = 45, b = 55, size = N)
    th2_0_d2_2_a2_3_th1_0 = 1 - th2_1_d2_2_a2_3_th1_0
    ### D2 = 3, A2 = 3
    th2_1_d2_3_a2_3_th1_0 = np.random.beta(a = 30, b = 70, size = N)
    th2_0_d2_3_a2_3_th1_0 = 1 - th2_1_d2_3_a2_3_th1_0


    param_arr = []
    for n in range(N):
        params = {}
        #####--------------------THETA1 =1  ------------------
        ### ---------------------NO DEFENSE-----------####
        params_d1 = {}
        #### NO ATTACK
        params_d1[1] = {'p0':1, 'p1':0}
        ### LOW ATTACK
        p1, p0 = th2_1_d2_1_a2_2_th1_1[n], th2_0_d2_1_a2_2_th1_1[n]
        params_d1[2] = {'p1':p1, 'p0':p0}
        ### HIGH ATTACK
        p1, p0 = th2_1_d2_1_a2_3_th1_1[n], th2_0_d2_1_a2_3_th1_1[n]
        params_d1[3] = {'p1':p1, 'p0':p0}


        ### ---------------------MINIMAL DEFENSE--------###
        params_d2 = {}
        #### NO ATTACK
        params_d2[1] = {'p0':1, 'p1':0}
        ### LOW ATTACK
        p1, p0 = th2_1_d2_2_a2_2_th1_1[n], th2_0_d2_2_a2_2_th1_1[n]
        params_d2[2] = {'p1':p1, 'p0':p0}
        ### HIGH ATTACK
        p1, p0 = th2_1_d2_2_a2_3_th1_1[n], th2_0_d2_2_a2_3_th1_1[n]
        params_d2[3] = {'p1':p1, 'p0':p0}



        ### ----------------------HIGH DEFENSE------------###
        params_d3 = {}
        #### NO ATTACK
        params_d3[1] = {'p0':1, 'p1':0}
        ### LOW ATTACK
        p1, p0 =  th2_1_d2_3_a2_2_th1_1[n], th2_0_d2_3_a2_2_th1_1[n]
        params_d3[2] = {'p1':p1, 'p0':p0}
        ### HIGH ATTACK
        p1, p0 = th2_1_d2_3_a2_3_th1_1[n], th2_0_d2_3_a2_3_th1_1[n]
        params_d3[3] = {'p1':p1, 'p0':p0}

        params[1] = {1:params_d1,2:params_d2,3:params_d3}


        #####----------------- THETA1 = 0 ---------------------
        ### ---------------------NO DEFENSE-----------####
        params_d1 = {}
        #### NO ATTACK
        params_d1[1] = {'p0':1, 'p1':0}
        ### LOW ATTACK
        p1, p0 = th2_1_d2_1_a2_2_th1_0[n], th2_0_d2_1_a2_2_th1_0[n]
        params_d1[2] = {'p1':p1, 'p0':p0}
        ### HIGH ATTACK
        p1, p0 = th2_1_d2_1_a2_3_th1_0[n], th2_0_d2_1_a2_3_th1_0[n]
        params_d1[3] = {'p1':p1, 'p0':p0}


        ### ---------------------MINIMAL DEFENSE--------###
        params_d2 = {}
        #### NO ATTACK
        params_d2[1] = {'p0':1, 'p1':0}
        ### LOW ATTACK
        p1, p0 = th2_1_d2_2_a2_2_th1_0[n], th2_0_d2_2_a2_2_th1_0[n]
        params_d2[2] = {'p1':p1, 'p0':p0}
        ### HIGH ATTACK
        p1, p0 = th2_1_d2_2_a2_3_th1_0[n], th2_0_d2_2_a2_3_th1_0[n]
        params_d2[3] = {'p1':p1, 'p0':p0}



        ### ----------------------HIGH DEFENSE------------###
        params_d3 = {}
        #### NO ATTACK
        params_d3[1] = {'p0':1, 'p1':0}
        ### LOW ATTACK
        p1, p0 =  th2_1_d2_3_a2_2_th1_0[n], th2_0_d2_3_a2_2_th1_0[n]
        params_d3[2] = {'p1':p1, 'p0':p0}
        ### HIGH ATTACK
        p1, p0 = th2_1_d2_3_a2_3_th1_0[n], th2_0_d2_3_a2_3_th1_0[n]
        params_d3[3] = {'p1':p1, 'p0':p0}

        params[0] = {1:params_d1,2:params_d2,3:params_d3}

        param_arr.append(params)

    return param_arr


def pDtheta2():  #### CHECK

    """
    Generates the params to generate a sample pD(theta2|d1,a1,theta1). 
    
    Parameters
    ------------

    Returns:

    Array-like: 
        Dictionary containg the parameters for pD(theta2|d1,a1,theta1)
    
    """
    params = {}
    #####--------------------THETA1 =1  ------------------
    ### ---------------------NO DEFENSE-----------####
    params_d1 = {}
    #### NO ATTACK
    params_d1[1] = {'p0':1, 'p1':0}
    ### LOW ATTACK
    params_d1[2] = {'p0':0.10, 'p1':0.90}
    ### HIGH ATTACK
    params_d1[3] = {'p0':0, 'p1':1}


    ### ---------------------MINIMAL DEFENSE--------###
    params_d2 = {}
    #### NO ATTACK
    params_d2[1] = {'p0':1, 'p1':0}
    ### LOW ATTACK
    params_d2[2] = {'p0': 0.30, 'p1':0.70}
    ### HIGH ATTACK
    params_d2[3] = {'p0':0.10, 'p1':0.90}



    ### ----------------------HIGH DEFENSE------------###
    params_d3 = {}
    #### NO ATTACK
    params_d3[1] = {'p0':1, 'p1':0}
    ### LOW ATTACK
    params_d3[2] = {'p0':0.70, 'p1':0.30}
    ### HIGH ATTACK
    params_d3[3] = {'p0':0.40, 'p1':0.60}

    params[1] = {1:params_d1,2:params_d2,3:params_d3}


    #####----------------- THETA1 = 0 ---------------------
    ### ---------------------NO DEFENSE-----------####
    params_d1 = {}
    #### NO ATTACK
    params_d1[1] = {'p0':1, 'p1':0}
    ### LOW ATTACK
    params_d1[2] = {'p0':0.55, 'p1':0.45}
    ### HIGH ATTACK
    params_d1[3] = {'p0':0.50, 'p1':0.50}


    ### ---------------------MINIMAL DEFENSE--------###
    params_d2 = {}
    #### NO ATTACK
    params_d2[1] = {'p0':1, 'p1':0}
    ### LOW ATTACK
    params_d2[2] = {'p0':0.65, 'p1':0.35}
    ### HIGH ATTACK
    params_d2[3] = {'p0':0.55, 'p1':0.45}



    ### ----------------------HIGH DEFENSE------------###
    params_d3 = {}
    #### NO ATTACK
    params_d3[1] = {'p0':1, 'p1':0}
    ### LOW ATTACK
    params_d3[2] ={'p0':0.85, 'p1':0.15}
    ### HIGH ATTACK
    params_d3[3] = {'p0':0.70, 'p1':0.30}

    params[0] = {1:params_d1,2:params_d2,3:params_d3}


    return params

def pAd1_sample(N):    #### CHECK
    """
    Generates the params to generate N random probabily samples 
    PA(D1). 
    
    Parameters
    ------------
    N: Integer

    Returns:

    Array-like: 
        List with N dictionaries containg the parameters for pA(d1)
    
    """

    params_arr = []

    d1_sample = np.random.dirichlet((1,3999,6000),size = N)

    for n in range(N):
        params_arr.append(d1_sample[n])

    return params_arr




def pAd2_sample(N): ### CHECK
    """
    Generates the params to generate N random probabily samples 
    PA(D2|d1,a,theta). 
    
    Parameters
    ------------
    N: Integer

    Returns:

    Array-like: 
        List with N dictionaries containg the parameters for pA(d2|d1,a,theta)
    
    """

  

    d1_1_a1_2_th1_1 = np.random.dirichlet((1,999,9000),size = N)
    d1_1_a1_3_th1_1 = np.random.dirichlet((1,499,9500),size = N)
    d1_2_a1_2_th1_1 = np.random.dirichlet((1,399,9600),size = N)
    d1_2_a1_3_th1_1 = np.random.dirichlet((1,299,9700),size = N)
    d1_3_a1_2_th1_1 = np.random.dirichlet((1,199,9800),size = N)
    d1_3_a1_3_th1_1 = np.random.dirichlet((1,99,9900),size = N)
  
    
    d1_1_a1_1_th1_0 =  np.random.dirichlet((25,3775,6200),size = N)
    d1_1_a1_2_th1_0 = np.random.dirichlet((1,4499,4500),size = N)
    d1_1_a1_3_th1_0 = np.random.dirichlet((1,499,9500),size = N)
    d1_2_a1_1_th1_0 = np.random.dirichlet((1,4499,4500),size = N)
    d1_2_a1_2_th1_0 = np.random.dirichlet((4500,4500,1000),size = N)
    d1_2_a1_3_th1_0 = np.random.dirichlet((1,4499,4500),size = N)
    d1_3_a1_1_th1_0 = np.random.dirichlet((1,499,9500),size = N)
    d1_3_a1_2_th1_0 = np.random.dirichlet((8500,1000,500),size = N)
    d1_3_a1_3_th1_0 = np.random.dirichlet((7500,1500,1000),size = N)
  
    params_arr = []
    
    for n in range(N):
      params = {}
      ################# THETA1 = 1 ###############################
      ###### D1 = 1
      params_d1 = {}
      ### A1 = 1
      params_d1[1] = [-1,-1,-1]      
      ### A1 = 2
      params_d1[2] = d1_1_a1_2_th1_1[n]
      ### A1 = 3
      params_d1[3] = d1_1_a1_3_th1_1[n]
      ###### D1 = 2
      params_d2 = {}
      ### A1 = 1
      params_d2[1] = [-1,-1,-1]      
      ### A1 = 2
      params_d2[2] = d1_2_a1_2_th1_1[n]
      ### A1 = 3
      params_d2[3] = d1_2_a1_3_th1_1[n]
      ####### D1 = 3
      params_d3 = {}
      ### A1 = 1
      params_d3[1] = [-1,-1,-1]      
      ### A1 = 2
      params_d3[2] = d1_3_a1_2_th1_1[n]
      ### A1 = 3
      params_d3[3] = d1_3_a1_3_th1_1[n]

      params[1] = {1:params_d1,2:params_d2,3: params_d3}

      ################## THETA1 = 0 ########################## 

      ###### D1 = 1
      params_d1 = {}
      ### A1 = 1
      params_d1[1] = d1_1_a1_1_th1_0[n]  
      ### A1 = 2
      params_d1[2] = d1_1_a1_2_th1_0[n]
      ### A1 = 3
      params_d1[3] = d1_1_a1_3_th1_0[n]
      ###### D1 = 2
      params_d2 = {}
      ### A1 = 1
      params_d2[1] = d1_2_a1_1_th1_0[n]
      ### A1 = 2
      params_d2[2] = d1_2_a1_2_th1_0[n]
      ### A1 = 3
      params_d2[3] = d1_2_a1_3_th1_0[n]
      ####### D1 = 3
      params_d3 = {}
      ### A1 = 1
      params_d3[1] = d1_3_a1_1_th1_0[n]
      ### A1 = 2
      params_d3[2] = d1_3_a1_2_th1_0[n]
      ### A1 = 3
      params_d3[3] = d1_3_a1_3_th1_0[n]

      params[0] = {1:params_d1,2:params_d2,3: params_d3}

      params_arr.append(params)

    return params_arr

def UA_sample(N):   #### CHECK
  
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

    return np.random.uniform(0,15, size=N)
