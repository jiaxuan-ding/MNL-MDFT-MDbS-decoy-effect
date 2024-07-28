# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 19:34:34 2024

@author: jiaxu
"""

import numpy as np
import pandas as pd
import random as random
import math
from scipy.stats import multivariate_normal
import scipy as sp
from scipy.linalg import fractional_matrix_power
from scipy.stats import norm
from scipy import integrate
from pyarma import *
from scipy.stats import norm
from numpy.linalg import det

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
data=np.vstack((RC_ICEV,OC_ICEV,DR_ICEV,RC_EVA,OC_EVA,DR_EVA,RC_EVB,OC_EVB,DR_EVB)).T.reshape(N_id,N_scenarios,N_alt,n_att)

#normlize the attribute matrix
data_att_inn=np.empty((N_id,N_scenarios,N_alt,n_att))
data_att_inn[:,:,:,0]= (-(data[:,:,:,0]-np.min(data[:,:,:,0]))/(np.max(data[:,:,:,0])-np.min(data[:,:,:,0])) )  -0.01  
data_att_inn[:,:,:,1]= (-(data[:,:,:,1]-np.min(data[:,:,:,1]))/(np.max(data[:,:,:,1])-np.min(data[:,:,:,1]))   )  -0.01  
data[:,:,:,2]=np.log(data[:,:,:,2])
data_att_inn[:,:,:,2]= ((data[:,:,:,2]-np.min(data[:,:,:,2]))/(np.max(data[:,:,:,2])-np.min(data[:,:,:,2]))   )  +0.01  

#error
err=np.random.normal(size=500*N_id*N_scenarios*N_alt).reshape(500,N_id,N_scenarios,N_alt,1)


#Contarst martix:C
contrast_val = -1/(N_alt - 1)
C = np.full((N_alt, N_alt), contrast_val)
np.fill_diagonal(C, 1) 
C=C.reshape(1,1,N_alt,N_alt)

#attention weight:equal weighted
w_p=np.repeat(np.array(1/n_att),n_att).reshape(n_att,1)
x_list = np.array([0,1,2])
prob = np.array([1/3,1/3,1/3])
attention_shift=np.random.choice(a=x_list, size=500, replace=True, p=prob)


###################################### defining log-likelyhood fuction ##############################
#define the function to calculate distance matrix D
def calsqdistance(M):
    X=np.zeros((M.shape[0],M.shape[1],M.shape[2],M.shape[2]))
    for i in range(0,M.shape[2]-1):
        for j in range(i+1,M.shape[2]):
            X[:,:,i,j]=np.sum(((M[:,:,i,:]-M[:,:,j,:]))**2,axis=2)
            X[:,:,j,i]=np.sum(((M[:,:,j,:]-M[:,:,i,:]))**2,axis=2)
    return X


#define the prduct of feedback matrix 
def cals1(S,t):
    temp=np.zeros(([S.shape[0],S.shape[1],S.shape[2]]))
    for i in range(0,S.shape[0]):
        A=mat(S.shape[1],S.shape[2], fill.zeros)
        for k1 in range(0,S.shape[1]):
            for k2 in range(0,S.shape[2]):
                A[k1,k2]=S[i,k1,k2]
        eigval = mat()
        eigvec = mat()
        eig_sym(eigval, eigvec, A)
        SB=powmat(diagmat(eigval), t)
        OUT=eigvec*SB*eigvec.t()
        for k1 in range(0,S.shape[1]):
            for k2 in range(0,S.shape[2]):
                temp[i,k1,k2]=np.real(OUT[k1,k2]) 
    return temp


#check if the covariance matrix is positive definite
def is_pos_def(x):
    A=mat(x.shape[0],x.shape[1], fill.zeros)
    for i in range(0,x.shape[0]):
        for j in range(0,x.shape[1]):
            A[i,j]=x[i,j]
    return A.is_sympd()
    

#calculate the CDF of the multivariate normal distribution
def check_p(mu1,ta1o,N_alt):
    if (np.sum(np.isnan(ta1o))!=0) or (np.sum(np.isnan(mu1))!=0) or (np.sum(~np.isfinite(ta1o))!=0) or (np.sum(~np.isfinite(mu1))!=0) or (np.sum(ta1o==np.Inf)!=0) or (np.sum(mu1==np.Inf)!=0):
        p=1e-50

    elif (np.sum(ta1o==0)==1) or (np.sum(np.diag(ta1o)<=0)!=0) or (np.sum(np.abs(mu1.flatten()))<0.0000000001) or (is_pos_def(ta1o)==False) or(np.abs(det(ta1o))<=0.001):
        p=1/N_alt
    else:
        p=multivariate_normal.cdf(x=mu1.flatten(),cov=ta1o,allow_singular=1)
    return p

#calculate the standard error of the transformed parameter
def trans(params,cov):
    sample_p=np.random.multivariate_normal(params, cov, 1000000)
    phi1 =1/(1+np.exp(-sample_p[:,0]));phi2=1/(1+np.exp(-sample_p[:,1]));t=(1+np.exp(sample_p[:,5]))
    return np.array([np.std(phi1),np.std(phi2),np.std(t)])

           
#formulate llh function
def MDFT_LL(param, C, data_att, w_p,Nid, Nscenario,N_alt,n_att,L):

    phi1 = 1/(1+np.exp(-param[0]))
    phi2 = 1/(1+np.exp(-param[1]))

    use_scaling=1
    M=data_att
    if use_scaling:
        M=M*param[2:5]
        
    else:
        w_p=(np.exp(param[2:5])/np.sum(np.exp(param[2:5]))).reshape(3,1)
        
    p_0=np.array([[0],[param[6]],[param[6]]]) #Intital preference

    Distsq=calsqdistance(M)
    epsilon=1
    
#feedback matrix: S
    S=np.eye(N_alt)-phi2*np.exp(-phi1*Distsq) 
    Sprob_E=np.zeros((Nid,Nscenario)) #choice probability
    
    t_step=1+np.exp(param[5]) 
    #param[5]
    #  # we do not estimate t in this code  
    #check coeff
    if t_step<1:
        t_step=1
    if phi1 < 1e-07:
        phi1 = 1e-07
    if np.abs(phi2) < 1e-07:
        phi2 = 0
    
    if np.abs(phi2) >= 0.999: 
        P = np.log(1e-50)*Nid*Nscenario 
        return -P
    elif t_step>1e4:
        return -np.log(1e-50)*Nid*Nscenario 
    # calculate Expectation
    ###check
    else:
        Scheck2 = (1-phi2) * N_alt+ 0.0000000001
        Scheck=np.sum(np.sum(np.abs(S),axis=3),axis=2) #or:Scheck1=np.sum(np.sum(np.abs(S),axis=2),axis=2)
        Scheck3=np.sum(np.sum(np.abs(Distsq),axis=3),axis=2) 
        
        mu=w_p.reshape((1,1,n_att,1)) #expectation of the weight matrix
        psi=(np.diag(w_p.flatten())-(w_p@w_p.T)).reshape(1,1,n_att,n_att)
        
        EP=np.zeros((M.shape[0],M.shape[1],M.shape[2],1))
        COVP=np.zeros((M.shape[0],M.shape[1],M.shape[2],M.shape[2]))
        
        twmp1=Scheck<Scheck2
        EP[twmp1]=(t_step*((C@M@mu)+ p_0))[twmp1]
        COVP[twmp1]=(t_step*(C@M@psi@M.transpose(0,1,3,2)@C.transpose(0,1,3,2)+(epsilon**2)*np.eye(N_alt)))[twmp1]
        
        twmp2=Scheck3<0.000001
        EP[twmp2]=(t_step*((C@M@mu)+p_0))[twmp2]
        COVP[twmp2]=(t_step*(C@M@psi@M.transpose(0,1,3,2)@C.transpose(0,1,3,2)+(epsilon**2)*np.eye(N_alt)))[twmp2]
        
        
        twmp3= ~((Scheck<Scheck2)+(Scheck3<0.000001))
        S=S[twmp3];M=M[twmp3]
        NS=cals1(S,t_step)
        EP[twmp3]=np.linalg.inv(np.eye(N_alt)-S)@(np.eye(N_alt)-NS)@(C@M@mu)+ NS@p_0
        
        Z=np.zeros((S.shape[0],N_alt*N_alt,N_alt*N_alt))
        Fai=C@M@psi@M.transpose(0,2,1)@C.transpose(0,1,3,2)+(epsilon**2)*np.eye(N_alt)
        FaiN=Fai.reshape((S.shape[0],N_alt*N_alt,1))
        for iid in range(0,S.shape[0]):
            Z[iid,:,:]=np.kron(S[iid,:,:],S[iid,:,:])
        NZ=cals1(Z,t_step) 
        COVP[twmp3]=(np.linalg.inv(np.eye(N_alt*N_alt)-Z)@(np.eye(N_alt*N_alt)-NZ)@FaiN).reshape(S.shape[0],N_alt,N_alt)


        ##probility
        TAO=L@EP #expectaion:(Nid,Nscenario,N_alt-1,1)
        sanjiao=L@COVP@L.transpose(0,1,3,2) #cov:(Nid,Nscenario,N_alt-1,N_alt-1)
        
        #multivariate normal distribution   
        for iid in range(0,Nid):
            for ic in range(0,Nscenario): 
                Sprob_E[iid,ic]=check_p(TAO[iid,ic,:],sanjiao[iid,ic,:,:],N_alt)
        Sprob_E[Sprob_E<=0.0001]=1e-50
        LLH=-np.sum(np.log(Sprob_E)) #log likelihood function #log likelihood function  
        return LLH

#calculate the probability     
def MDFT_genrate_choice(param, C, data_att,Nid, Nscenario,N_alt,n_att,attention_shift,err):
    # phi1 and phi2 is between 0 and 1
    phi1 = 1/(1+np.exp(-param[0]))
    phi2 = 1/(1+np.exp(-param[1]))
    #attribute matrix
    M=data_att*param[2:5]
    #intial preference
    p_0=np.array([[0],[param[6]],[param[6]]])

    Distsq=calsqdistance(M)
    epsilon=1
    #feedback matrix: S
    S=np.eye(N_alt)-phi2*np.exp(-phi1*Distsq) 
    t_step=np.round(1+np.exp(param[5]),0) 
    #param[5] #
    
    attention=np.eye(3)
    V=np.empty((np.int64(t_step)+1,Nid, Nscenario,N_alt,1))
    V[0]=p_0
    for i in range(1,np.int64(t_step)+1):
        testexp=((C@M)@attention[attention_shift[i-1]]).reshape(Nid, Nscenario,N_alt,1)+err[i-1]
        V[i]=(S@V[i-1]+testexp)
    choice=np.argmax(V[np.int64(t_step)].reshape(Nid,Nscenario,N_alt),axis=2)    
    
    #for i in range(1,np.int64(t_step)+1):
        #print(i,np.sum(np.argmax(V[i].reshape(Nid, Nscenario,N_alt),axis=2)==1),np.sum(np.argmax(V[i].reshape(Nid, Nscenario,N_alt),axis=2)==2))
    
    return choice

#define the callback fuction of the estimation    
def callbackF(Xi):
    global Nfeval
    print ('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}   {6: 3.6f}    {7: 3.6f}     {8: 3.6f}    '.format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3],Xi[4], Xi[5], Xi[6],MDFT_LL(Xi,C, data_att_inn, w_p,N_id, N_scenarios,N_alt,n_att,L)))
    Nfeval += 1
 

###################################### generate choice ##############################
True_param=np.array([0.027, -1.544,  2.397,  3.18 ,  2.52 ,  3.485, -0.551])
                     #phi1,phi2,beta_rc,beta_oc,beta_dr,t,asc_ev
y_choice=MDFT_genrate_choice(True_param, C, data_att_inn,N_id, N_scenarios,N_alt,n_att,attention_shift,err) #the chosen alternative
    
L=np.zeros((N_id,N_scenarios,N_alt-1,N_alt))
for i1 in range(0,N_id):
    for i2 in range(0,N_scenarios):
        i4=0
        for i3 in range(0,N_alt):
            if i3==y_choice[i1,i2]:
                L[i1,i2,:,y_choice[i1,i2]]=np.ones(N_alt-1) #1s for chosen alternative (largest preference value)
            else:
                L[i1,i2,i4,i3]=-1
                i4+=1   #negative identity martix for others

tru_llh=MDFT_LL(True_param, C, data_att_inn, w_p,N_id, N_scenarios,N_alt,n_att,L) #True LLH Value

################################# estimation #############################
bounds=((-3,3),(-3,3),(0.1,20),(0.1,20),(0.1,20),(-3,5),(-5,5))


intial=np.array([1,1,3,3,3,1,1])
Nfeval = 1  
print('Iteration','\u03C6_1         ','\u03C6_2       ','\u03B2_RC         ','\u03B2_OC      ','\u03B2_DR      ','t       ','intercept.EV  ','Log-Likelihood')      

resOpt = sp.optimize.minimize(
                fun = MDFT_LL,
                x0 = intial,
                args = (C, data_att_inn, w_p,N_id, N_scenarios,N_alt,n_att,L),
                method = 'L-BFGS-B',
                bounds=bounds,
                tol=0.001,
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

EST=resOpt['x'].copy()#Estimtaed parameter
EST[0] = 1/(1+np.exp(-resOpt['x'][0]));EST[1] = 1/(1+np.exp(-resOpt['x'][1]));EST[5] = 1+np.exp(resOpt['x'][5])
ERR=(np.sqrt(np.diag(resOpt.hess_inv.todense()))).copy() #Standard erro
ERR[[0,1,5]]=trans(resOpt['x'],resOpt.hess_inv.todense())
Z_VAL=EST/ERR #z-value
P_VAL=2*norm.cdf(-np.abs(Z_VAL)) #p-value
SIG=trans_significance(P_VAL) #singificant level
        
###########################################return results########################################
info = f"""
Estimation summary
------------------------------------------------------------------------------------
Coefficient         Estimate      Std.Err.       z-val         P>|z|
------------------------------------------------------------------------------------
\u03C6_1               {EST[0]:10.6f}   {ERR[0]:10.6f}     {Z_VAL[0]:10.6f}     {P_VAL[0]:10.6f} {SIG[0]}
\u03C6_2               {EST[1]:10.6f}   {ERR[1]:10.6f}     {Z_VAL[1]:10.6f}     {P_VAL[1]:10.6f} {SIG[1]}
\u03B2_RC              {EST[2]:10.6f}   {ERR[2]:10.6f}     {Z_VAL[2]:10.6f}     {P_VAL[2]:10.6f} {SIG[2]}
\u03B2_OC              {EST[3]:10.6f}   {ERR[3]:10.6f}     {Z_VAL[3]:10.6f}     {P_VAL[3]:10.6f} {SIG[3]}
\u03B2_DR              {EST[4]:10.6f}   {ERR[4]:10.6f}     {Z_VAL[4]:10.6f}     {P_VAL[4]:10.6f} {SIG[4]}
t                 {EST[5]:10.6f}   {ERR[5]:10.6f}     {Z_VAL[5]:10.6f}     {P_VAL[5]:10.6f} {SIG[5]}
intercept.EV      {EST[6]:10.6f}   {ERR[6]:10.6f}     {Z_VAL[6]:10.6f}     {P_VAL[6]:10.6f} {SIG[6]}
------------------------------------------------------------------------------------
Significance:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Log-Likelihood={-resOpt['fun']:10.4f}
BIC={len(True_param)*np.log(N_id*N_scenarios)+2*resOpt['fun']:10.4f}

Log-Likelihood recovery:
Ture Log-Likelihood={-tru_llh:10.4f}
Difference in Log-Likelihood={np.abs(-tru_llh+resOpt['fun']):10.4f}
Relative difference in Log-Likelihood={np.abs(-tru_llh+resOpt['fun'])/tru_llh:10.4f}
"""

# Print the formatted information
print(info)

    

