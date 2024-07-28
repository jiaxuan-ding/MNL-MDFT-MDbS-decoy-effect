# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 21:14:37 2024

@author: jiaxu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opts
import random
import math
from numpy.linalg import inv
from scipy.stats import norm
import scipy as sp

###################################### generate data #########################################
N_id= 18  #number of respondents
N_scenarios=4 #number of scenarios each respondent answered
N_alt=3 #numner of alternatives:3
n_att=3 #number of attributes：3
total=N_id*N_scenarios #total number of scenarios

#set seeds
np.random.seed(140)
#attribute value of ICEV (competitor)
daily_diatance=np.random.uniform(1.5,3,N_id)
RC_ICEV=np.repeat(np.random.uniform(1800,2200,N_id),N_scenarios) #monthly renting cost of ICEV
OC_ICEV=np.repeat(np.random.uniform(13,18,N_id)*daily_diatance,N_scenarios)  #daily operating cost of ICEV= price*daily distance
DR_ICEV=np.repeat(np.random.uniform(600,800,N_id),N_scenarios) #driving range of ICEV

#attribute value of EVA (target)： equally attractive 
RC_EVA=RC_ICEV*np.repeat([np.array([1.2,1.1,1,0.9,0.8])[i] for i in np.random.randint(0,5,N_id)],N_scenarios) #Monthly renting cost 
OC_EVA=np.repeat(daily_diatance*6,N_scenarios)  #daily operating cost
DR_EVA=DR_ICEV*0.6 #driving range 

#attribute value of EVB (decoy)： 
RC_EVB=RC_EVA-[np.array([20,50])[i] for i in np.random.randint(0,2,total)] #Monthly renting cost 
OC_EVB=OC_EVA  #daily operating cost 
DR_EVB=DR_EVA -[np.array([50,100,150])[i] for i in np.random.randint(0,3,total)]#driving range

#attribute matrix
data=np.vstack((RC_ICEV,OC_ICEV,DR_ICEV,RC_EVA,OC_EVA,DR_EVA,RC_EVB,OC_EVB,DR_EVB)).T.reshape(total,N_alt,n_att)

#normlize the attribute matrix
data_att_inn=np.empty((total,N_alt,n_att))
data_att_inn[:,:,0]= (-(data[:,:,0]-np.min(data[:,:,0]))/(np.max(data[:,:,0])-np.min(data[:,:,0])) )  -0.01  
data_att_inn[:,:,1]= (-(data[:,:,1]-np.min(data[:,:,1]))/(np.max(data[:,:,1])-np.min(data[:,:,1]))   )  -0.01  
data[:,:,2]=np.log(data[:,:,2])
data_att_inn[:,:,2]= ((data[:,:,2]-np.min(data[:,:,2]))/(np.max(data[:,:,2])-np.min(data[:,:,2]))   )  +0.01  

#error
err=np.random.gumbel(size=total*N_alt).reshape(total,N_alt)

###################################### defining log-likelyhood fuction ##############################
#LLH fuction
def MNL(param,dataset,decoy_choice,n_gen,refer_var):
    ASC=np.insert(np.repeat(param[0],2),refer_var,[0]).reshape(1,3)
    V=np.exp(np.sum(param[1:1+n_gen]*dataset,axis=2)+ASC)
    P=V/np.sum(V,axis=1,keepdims=True)
    MLL=decoy_choice*np.log(P)
    return -np.sum(MLL)

#generate the choice    
def MNL_gen(param,dataset,n_gen,refer_var,err):
    ASC=np.insert(np.repeat(param[0],2),refer_var,[0]).reshape(1,3)
    V=np.exp(np.sum(param[1:1+n_gen]*dataset,axis=2)+ASC+err)
    argmaxv=np.argmax(V,axis=1)
    choice=np.zeros((dataset.shape[0],dataset.shape[1]))
    for i in range(argmaxv.shape[0]):
        choice[i, argmaxv[i]]=1
    return choice


###################################### setting true parameter values ##############################
True_beta=np.array([0.819, 11.974, 10.173,  6.543]) #true parameter values:ASC_EV,beta_rc,beta_oc,bets_dr
y_choice=MNL_gen(True_beta,data_att_inn,3,0,err)#chosen alternatives
tru_llh=MNL(True_beta,data_att_inn,y_choice,3,0) #True LLH Value

###################################### estimating MNL Model ##############################
def callbackF(Xi):
    global Nfeval
    print ('{0:4d}     {1: 3.6f}    {2: 3.6f}     {3: 3.6f}     {4: 3.6f}     {5: 3.6f} '.format(Nfeval,  Xi[0], Xi[1], Xi[2], Xi[3], MNL(Xi,data_att_inn,y_choice,3,0)))
    Nfeval += 1
    
intial_beta=np.ones(len(True_beta))
Nfeval=1

print('Iteration','intercept.EV  ','\u03B2_RC        ','\u03B2_OC       ','\u03B2_DR        ','Log-Likelihood')
resOpt = sp.optimize.minimize(
        fun = MNL,
        x0 = intial_beta,
        args = (data_att_inn,y_choice,3,0),
        method = 'BFGS',#L-BFGS-B    
        #bounds=bnds1,
        tol=0.0001,
        callback=callbackF,
        options = {'disp': True}
        )


#Translates a p-value into a significance level based on predefined thresholds.
def trans_significance(p_value):
    sig_level=[]
    for i in p_value:
        if i < 0.001:
            sig_level.append('***')
        elif i < 0.01:
            sig_level.append('**')
        elif i < 0.05:
            sig_level.append('*')
        elif i < 0.1:
            sig_level.append('.')
        else:
            sig_level.append(' ')
    return sig_level

ERR=np.sqrt(np.diag(resOpt['hess_inv'])) #Standard erro
Z_VAL=resOpt['x']/np.sqrt(np.diag(resOpt['hess_inv'])) #z-value
P_VAL=2*norm.cdf(-np.abs(Z_VAL)) #p-value
SIG=trans_significance(P_VAL) #singificant level
        
###########################################return results########################################
info = f"""
Estimation summary
------------------------------------------------------------------------------------
Coefficient          Estimate      Std.Err.       z-val         P>|z|
------------------------------------------------------------------------------------
intercept.EV       {resOpt['x'][0]:10.6f}   {ERR[0]:10.6f}     {Z_VAL[0]:10.6f}     {P_VAL[0]:10.6f} {SIG[0]}
\u03B2_RC               {resOpt['x'][1]:10.6f}   {ERR[1]:10.6f}     {Z_VAL[1]:10.6f}     {P_VAL[1]:10.6f} {SIG[1]}
\u03B2_OC               {resOpt['x'][2]:10.6f}   {ERR[2]:10.6f}     {Z_VAL[2]:10.6f}     {P_VAL[2]:10.6f} {SIG[2]}
\u03B2_DR               {resOpt['x'][3]:10.6f}   {ERR[3]:10.6f}     {Z_VAL[3]:10.6f}     {P_VAL[3]:10.6f} {SIG[3]}
------------------------------------------------------------------------------------
Significance:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Log-Likelihood={-resOpt['fun']:10.4f}
BIC={len(intial_beta)*np.log(N_id*N_scenarios)+2*resOpt['fun']:10.4f}


Log-Likelihood recovery:
Ture Log-Likelihood={-tru_llh:10.4f}
Difference in Log-Likelihood={np.abs(-tru_llh+resOpt['fun']):10.4f}
Relative difference in Log-Likelihood={np.abs(-tru_llh+resOpt['fun'])/tru_llh:10.4f}
"""

# Print the formatted information
print(info)
