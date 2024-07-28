# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 23:46:00 2024

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
data=np.vstack((RC_ICEV,OC_ICEV,DR_ICEV,RC_EVA,OC_EVA,DR_EVA,RC_EVB,OC_EVB,DR_EVB)).T.reshape(N_id,N_scenarios,N_alt,n_att)

#normlize the attribute matrix
data_att_inn=np.empty((N_id,N_scenarios,N_alt,n_att))
data_att_inn[:,:,:,0]= (-(data[:,:,:,0]-np.min(data[:,:,:,0]))/(np.max(data[:,:,:,0])-np.min(data[:,:,:,0])) )  -0.01  
data_att_inn[:,:,:,1]= (-(data[:,:,:,1]-np.min(data[:,:,:,1]))/(np.max(data[:,:,:,1])-np.min(data[:,:,:,1]))   )  -0.01  
data[:,:,:,2]=np.log(data[:,:,:,2])
data_att_inn[:,:,:,2]= ((data[:,:,:,2]-np.min(data[:,:,:,2]))/(np.max(data[:,:,:,2])-np.min(data[:,:,:,2]))   )  +0.01  



###################################### defining log-likelyhood fuction #############################
#filling the diagonal elements with na value
def fill_dignan(data):
    for i in range(0,data.shape[0]):
        for j in range(data.shape[1]):
            row, col = np.diag_indices_from(data[i,j]) 
            (data[i,j])[row, col] = np.nan
    return data

#probability of accumulating evidence towards alternatives
def cal_accumu_p(tdata,beta):
    sc=beta[0]
    beta_0=(np.repeat(0.7/(1+np.exp(beta[1:4])),tdata.shape[3])).reshape(1,1,9,1)
    beta_1=-110/(1+np.exp(beta[4]))
    
    ASC_EVA=beta[5]
    ASC=np.array([0,-ASC_EVA,-ASC_EVA,ASC_EVA,0,0,ASC_EVA,0,0]).reshape(3,3)
    ASC1=np.zeros((9,9));ASC1[0:3,0:3]=ASC;ASC1[3:6,3:6]=ASC;ASC1[6:9,6:9]=ASC
    ASC1=ASC1.reshape(1,1,9,9)
    output=np.full([tdata.shape[0],tdata.shape[1],tdata.shape[2]*tdata.shape[3],tdata.shape[2]*tdata.shape[3]],np.nan) #distance matrix
    for i in range(0,tdata.shape[3]):
        output[:,:,tdata.shape[2]*i:tdata.shape[2]*i+tdata.shape[2],tdata.shape[2]*i:tdata.shape[2]*i+tdata.shape[2]]=np.absolute((np.repeat(tdata[:,:,:,i],tdata.shape[2])).reshape(tdata.shape[0],tdata.shape[1],tdata.shape[2],tdata.shape[2])-(np.tile(tdata[:,:,:,i],tdata.shape[2])).reshape(tdata.shape[0],tdata.shape[1],tdata.shape[2],tdata.shape[2]))/np.absolute(np.tile(tdata[:,:,:,i],tdata.shape[2])).reshape(tdata.shape[0],tdata.shape[1],tdata.shape[2],tdata.shape[2])     
    output=fill_dignan(output)
    simi_dis=np.full([tdata.shape[0],tdata.shape[1],tdata.shape[2]*tdata.shape[3],tdata.shape[2]*tdata.shape[3]],np.nan) #similarity matrix
    simi_dis[~np.isnan(output)]=np.exp(-sc*output[~np.isnan(output)]) 

    p_evalulate=np.nansum(simi_dis,axis=3)/np.nansum(np.nansum(simi_dis,axis=2),axis=2,keepdims=1)
    matrix_g=1/(1+np.exp(beta_1*(output-beta_0)+ASC1))
    for i in range(0,tdata.shape[3]):
        temp1=((np.repeat(tdata[:,:,:,i],tdata.shape[2])).reshape(tdata.shape[0],tdata.shape[1],tdata.shape[2],tdata.shape[2])-(np.tile(tdata[:,:,:,i],tdata.shape[2])).reshape(tdata.shape[0],tdata.shape[1],tdata.shape[2],tdata.shape[2]))<=0
        (matrix_g[:,:,tdata.shape[2]*i:tdata.shape[2]*i+tdata.shape[2],tdata.shape[2]*i:tdata.shape[2]*i+tdata.shape[2]])[temp1]=0
    matrix_g=fill_dignan(matrix_g)
    p_wincomp=np.nanmean(matrix_g,axis=3)
    p_accum=np.sum((p_wincomp*p_evalulate).reshape(p_wincomp.shape[0],p_wincomp.shape[1],tdata.shape[3],tdata.shape[2]),axis=2)
    return  p_accum

#total number of states
def compute_n_states(n_alternatives_,Range_):
    num=1; #Permutation :P(N_range+n_alt-2,N_range-1)
    den=1; #Factorial:(n_alt)!
    tmp=Range_[1]-Range_[0]-1;
    num=(math.factorial(n_alternatives_-2+tmp)/math.factorial(tmp-1))
    den=math.factorial(n_alternatives_-1)
    return int(num/den)        

#updating states
def update_states(state_,Range_,n_alternatives_):
    updated=False;
    i=0
    while(i<n_alternatives_-1): #only n_alterantive-1 is needed to be updated
        if updated:
            break
        elif state_[i]+1<Range_[1]: #if the updated state is still less than the upper bound 
            state_[i]=state_[i]+1; #satae update 1 
            updated=True           #do not update again
        else: #if the update 1 to state will equal the upper level
            state_[i]=Range_[0]+1 #state set to the lower vale+1
        i=i+1
    # at lesat one sate less than the upper bound
    if updated:
        j=0
        total=0
        while (j < len(state_)-1):
            total=total+state_[j]
            j=j+1
        negsum=-1*total
        if Range_[0]<negsum and negsum<Range_[1]:
            state_[n_alternatives_-1]=negsum;
        else:
            updated=update_states(state_,Range_,n_alternatives_)
            
    return updated



#LLH
def MDbS(beta,theta,data_att,y_choice):
    p_accum=cal_accumu_p(data_att,beta)
    p_total=np.zeros((p_accum.shape[0],p_accum.shape[1],p_accum.shape[2]))
    Time_total=np.zeros((p_accum.shape[0],p_accum.shape[1],p_accum.shape[2]))
    
    boundary=math.ceil(theta*data_att.shape[2])
    Range=[-1*(boundary+(data_att.shape[2]-2)*(boundary-1)), boundary]
    n_transient_states=compute_n_states(data_att.shape[2],Range)
    
    
    if n_transient_states<600:    
        state=[Range[0]]*data_att.shape[2]
        transient_state_index=[]
        i = 0;
        while update_states(state, Range, data_att.shape[2]):
               transient_state_index.append(list(state))
    else:
        print('too many states')

    for i1 in range(0,p_accum.shape[0]):
        for j1 in range(0,p_accum.shape[1]):
            Q=np.eye(n_transient_states)
            R=np.zeros((n_transient_states, data_att.shape[2]))
            Z=np.zeros((1,n_transient_states))
            I=np.eye(n_transient_states)
            one_matrix=np.ones((n_transient_states,1))
            Z[0,transient_state_index.index([0]*data_att.shape[2])]=1
            Q=(1-np.sum(p_accum[i1,j1]))* Q

            i=0
            while(i<len(transient_state_index)):    
                j=0
                while(j<data_att.shape[2]):
                    state_temp=transient_state_index[i].copy()
                    state_temp[j]=state_temp[j]+data_att.shape[2]
                    k=0
                    while(k<data_att.shape[2]): #every elements in state_temp minus 1
                        state_temp[k]=state_temp[k]-1
                        k=k+1
                    #2,-1,-1 or -1,2,-1 or -1,-1,2
                    if state_temp[j]<boundary:
                        Q[i, transient_state_index.index(state_temp)] = p_accum[i1,j1,j]

                    elif state_temp[j]>=boundary:
                        R[i,j]=p_accum[i1,j1,j]
                    j=j+1
                i=i+1

            IQ = I - Q
            if n_transient_states < 1000:
                IQinv = inv(IQ)
            else:
                IQinv = IQ.ldlt().solve(I)
                relative_error = (IQ * IQinv - I).norm() / I.norm();
                if relative_error > 1e-8:
                    raise Exception('The relative error is too large')
     
            ZIQinv = np.dot(Z,IQinv)
            p_total[i1,j1,:]= np.dot(ZIQinv,R)[0]

    p_total[p_total==0]=10**(-50)
    
    return -np.sum(np.log(np.sum(p_total*y_choice,axis=2)))


#defining the callback fuction of the estimation
def trans(params,cov):
    sample_p=np.random.multivariate_normal(params, cov, 1000000)
    beta_0_rc=0.7/(1+np.exp(sample_p[:,1]))
    beta_0_oc=0.7/(1+np.exp(sample_p[:,2]))
    beta_0_dr=0.7/(1+np.exp(sample_p[:,3]))
    beta_1=-110/(1+np.exp(sample_p[:,4]))
    return np.std(beta_0_rc),np.std(beta_0_oc),np.std(beta_0_dr),np.std(beta_1)

#defining the callback fuction of the estimation
def callbackF(Xi):
    global Nfeval
    global theta1
    print ('{0:4d}     {1: 3.6f}    {2: 3.6f}     {3: 3.6f}     {4: 3.6f}     {5: 3.6f}      {6: 3.6f}       {7: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3],Xi[4],Xi[5], MDbS(Xi,theta1,data_att_inn,y_choice)))
    Nfeval += 1

################## Genrate choice#####################################################
True_param=np.array([0.493, -0.521,  0.11 ,  1.967, -0.938, -0.856,  1.1])
                   #sc,beta_0_rc,beta_0_oc,beta_0_dr,beta_1,asc_ev,theta
                   
p_accum=cal_accumu_p(data_att_inn,True_param)
boundary=math.ceil(True_param[-1]*data_att_inn.shape[2])
x_list = np.array([0,1,2,3])
y_choice=np.zeros((p_accum.shape[0],p_accum.shape[1],3))
for i1 in range(0,p_accum.shape[0]):
    for j1 in range(0,p_accum.shape[1]):
        prob = np.hstack((p_accum[i1,j1],1-np.sum(p_accum[i1,j1])))
        n_icev=0;n_eva=0;n_evb=0
        while np.max(np.array([n_icev,n_eva,n_evb]))<3:
            attention_shift=np.random.choice(a=x_list, size=1, replace=True, p=prob)
            if attention_shift==0:
                n_icev+=1
            if attention_shift==1: 
                n_eva+=1
            if attention_shift==2:
                n_evb+=1
        y_choice[i1,j1,np.argmax(np.array([n_icev,n_eva,n_evb]))]=1
                   
tru_llh=MDbS(True_param,True_param[-1],data_att_inn,y_choice) #true LLH 
  
###########################eatimation###################################################      
bounds = ((0.1,np.Inf),(-5,5),(-5,5),(-5,5), (-5,3),(-3,3))  #bound
Nfeval = 1
intial=np.array([0.5,-0.5,0.5,0.5,0.5,1])
                #sc,beta_0_rc,beta_0_oc,beta_0_dr,beta_1,asc_ev,                
theta1=1.1  #theta
print('Iteration  ','\u03B1          ','\u03B2_0,RC      ','\u03B2_0,OC       ','\u03B2_0,DR        ','\u03B2_ 1      ','intercept.EV  ','Log-Likelihood')
resOpt = sp.optimize.minimize(
                fun = MDbS,
                x0 = intial,
                args = (theta1,data_att_inn,y_choice),
                method = 'L-BFGS-B',
                tol=0.001,
                bounds=bounds,
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
EST[[1,2,3]] = 0.7/(1+np.exp(-resOpt['x'][[1,2,3]]));EST[4] = -110/(1+np.exp(-resOpt['x'][4]))
ERR=(np.sqrt(np.diag(resOpt.hess_inv.todense()))).copy() #Standard erro
ERR[[1,2,3,4]]=trans(resOpt['x'],resOpt.hess_inv.todense())
Z_VAL=EST/ERR #z-value
P_VAL=2*norm.cdf(-np.abs(Z_VAL)) #p-value
SIG=trans_significance(P_VAL) #singificant level
        
###########################################return results########################################
info = f"""
Estimation summary
------------------------------------------------------------------------------------
Coefficient         Estimate      Std.Err.       z-val         P>|z|
------------------------------------------------------------------------------------
\u03B1               {EST[0]:10.6f}   {ERR[0]:10.6f}     {Z_VAL[0]:10.6f}     {P_VAL[0]:10.6f} {SIG[0]}
\u03B2_0,RC          {EST[1]:10.6f}   {ERR[1]:10.6f}     {Z_VAL[1]:10.6f}     {P_VAL[1]:10.6f} {SIG[1]}
\u03B2_0,OC          {EST[2]:10.6f}   {ERR[2]:10.6f}     {Z_VAL[2]:10.6f}     {P_VAL[2]:10.6f} {SIG[2]}
\u03B2_0,DR          {EST[3]:10.6f}   {ERR[3]:10.6f}     {Z_VAL[3]:10.6f}     {P_VAL[3]:10.6f} {SIG[3]}
\u03B2_ 1            {EST[4]:10.6f}   {ERR[4]:10.6f}     {Z_VAL[4]:10.6f}     {P_VAL[4]:10.6f} {SIG[4]}
intercept.EV    {EST[5]:10.6f}   {ERR[5]:10.6f}     {Z_VAL[5]:10.6f}     {P_VAL[5]:10.6f} {SIG[5]}
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

    
