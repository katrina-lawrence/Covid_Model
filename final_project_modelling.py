# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 19:17:38 2022

@author: Katrina Lawrence
"""

'''This file is to fit the n parameter SIR models (n= 2,4,5). This file will
have a bunch of output so if it's taking too long and you'd rather just check
one particular file then the code will still work if you copy/paste the 
data-processing (with imports) and individual model fitting piece you need 
into a new file. '''


'''Imports'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
from scipy import interpolate as inter
from scipy.integrate import odeint
from sklearn.metrics import r2_score as r2


''' Data pre-processing '''

''' Read Data'''

df = pd.read_csv('datathon_data.csv')

cases = df["new_cases_14_days"][431:645] #time interval (see excel sheet for dates)

t = np.arange(0,len(cases) ) #time (discrete)

#Visualize Data
plt.plot(t, cases, label = 'Observed Data') 
plt.title('Covid-19 Pandemic in Spain')
plt.xlabel('Time (days)')
plt.ylabel('Active Cases')
plt.show()

t_p = np.linspace(0, len(cases), len(cases)*10) #time 'continuous'

''' Fit of the 2-parameter model (IC's not fitted)'''

''' Set up Functions '''
# Set up ode solver with 'inline' function definition
def solve2(p,time, y0):
    '''
    

    Parameters
    ----------
    p : TYPE: List
        A list of the parameters to be fitted (Beta, Gamma)
    time : TYPE: Numpy Array
        
    y0 : List
        Initial conditon

    Returns
    -------
    Numerical solutions to the SIR system

    '''
    beta = p[0]
    gamma = p[1]
    sys = lambda u,t: [-1*beta*u[0]*u[1], beta*u[0]*u[1] - gamma*u[1] , gamma*u[1]]
    result = odeint(sys, y0,time)
    return(result)


#same params as solve2
def get_sse2(p,time,y0):
    tp = np.linspace(min(time), max(time), 10*len(time))
    obs = np.array(cases)[min(time):max(time)+1]
    result = solve2(p, tp,y0)
    f = inter.interp1d(tp, result[:, 1])
    i_new = f(time)
    resid = obs - i_new
    return (sum(resid ** 2))


  



'''Time 1'''

t1_interval = [0, 180]
t1 = np.arange(min(t1_interval), max(t1_interval))
t1p = np.linspace(min(t1), max(t1), 10*len(t1))
cases1 = np.array(cases)[min(t1):max(t1)+1]

#set parameter guesses
n = 25.69e6
i0 = 150
beta1 = 2e-9
gamma1 = 1/15.2
mu1 = 0.006
y0 = [n-i0-1000, i0, 1000]
params1 = [beta1,gamma1]


# Solve ode(s)
result1 = solve2(params1,t1p,y0)
labels = ['T(t)','I(t)']





#test other guesses

p1 = [1e-9, 1/14]
p2 = [3e-9, 1/15]

sse = [get_sse2(params1,t1,y0), get_sse2(p1, t1,y0), get_sse2(p2, t1,y0)]
### Print outputs of sse for changes in parameters (using f' strings again like in assignment 2 code)
print(f'''
p0 sse = {get_sse2(params1, t1,y0)}
p1 sse = {get_sse2(p1,t1,y0)}
p2 sse = {get_sse2(p2,t1,y0)}
min sse = {min(sse)}''')


'''Careful with fmin method especially the args parameter, don't forget 
the extra comment when you come back to this'''

soln1 = opt.fmin(get_sse2, p1, args=(t1,y0,), maxiter = 1000)
print(soln1) #optional
best_result = solve2(soln1, t1p,y0) #best fit
plt.plot(t1,cases1, 'b')
plt.plot(t1p, best_result[:,1], 'r')
plt.title('Soln to t1')
plt.show()
'''AIC'''
sse21 = get_sse2(soln1, t1,y0)
aic21 = len(t1)*np.log(sse21/len(t1))+2*len(soln1)

'''Time 2'''

t2_interval = [180, 213]
t2 = np.arange(min(t2_interval), max(t2_interval))
t2p = np.linspace(min(t2), max(t2), 10*len(t2))
cases2 = np.array(cases)[min(t2):max(t2)+1]

#set parameter guesses
n = 25.69e6
i1 = cases2[0]
r1 = sum(cases1)
beta2 = 2e-9
gamma2 = 1/15.2
mu2 = 0.006
y1 = [n-i0-r1, i1, r1]
params2 = [beta2,gamma2]


# Solve ode(s)
result2 = solve2(params2,t2p,y0)
labels = ['T(t)','I(t)']

### Show initial plot of obs data and inital model:

plt.xlabel('Time (days)', weight='bold')
plt.title('Solution to Model with Initial Guesses', weight='bold')
plt.ylabel('# of People', weight='bold')



plt.plot(t2,cases2, 'b')
plt.plot(t2p, result2[:,1], 'r')
# plt.plot(t, model)
plt.show()

#test other guesses

p1 = [3e-10, 1/14]
p2 = [1e-8, 1/20]

sse = [get_sse2(params2,t2,y1), get_sse2(p1, t2,y1), get_sse2(p2, t2,y1)]
### Print outputs of sse for changes in parameters (using f' strings again like in assignment 2 code)
print(f'''
p0 sse = {get_sse2(params2, t2,y1)}
p1 sse = {get_sse2(p1,t2,y1)}
p2 sse = {get_sse2(p2,t2,y1)}
min sse = {min(sse)}''')

soln2 = opt.fmin(get_sse2, p2, args=(t2,y1,), maxiter = 1000)
print(soln2)
best_result2 = solve2(soln2, t2p,y1)
plt.plot(t2,cases2, 'b')
plt.plot(t2p, best_result2[:,1], 'r')
plt.title('Soln to t2')
plt.show()

'''Get AIC (12 indicates 1st model 2nd time interval'''
sse22= get_sse2(soln2, t2,y1)
aic22 = len(t2)*np.log(sse22/len(t2)) + 2*len(soln2)

#VISUALATION
plt.plot(t, cases,'ko', label = 'Observed Data')
plt.title('Fit of 2-Parameter Model', fontweight='bold', fontsize = 15, pad=10)
plt.xlabel('Time (days)', fontweight='bold', fontsize= 15, labelpad=10)
plt.ylabel('Active Cases', fontweight='bold', fontsize = 15, labelpad=10)
plt.plot(t1p, best_result[:,1], 'c',linewidth=4, label = 'Fitted Model for Time 1')
plt.plot(t2p, best_result2[:,1],'r', linewidth =4,label = 'Fitted Model for Time 2' )
plt.legend(prop=dict(weight='bold', size = 13))
plt.xticks(fontweight='bold', fontsize = 13)
plt.yticks(fontweight='bold', fontsize = 13)
plt.show()


''' 4-parameter fit  (everything except gamma'''


''' Set up Functions '''
# Set up ode solver with 'inline' function definition
def solve4(p,time):
    beta = abs(p[0])
    gamma = 1/15.2
    s= abs(p[1])
    i = abs(p[2])
    r = abs(p[3])
    
    sys = lambda u,t: [-1*beta*u[0]*u[1], beta*u[0]*u[1] - gamma*u[1], gamma*u[1]]
    result = odeint(sys, [s, i, r],time)
    return(result)




def get_sse4(p,time):
    tp = np.linspace(min(time), max(time), 10*len(time))
    obs = np.array(cases)[min(time):max(time)+1]
    result = solve4(p, tp)
    f = inter.interp1d(tp, result[:, 1])
    i_new = f(time)
    resid = obs - i_new
    return (sum(resid ** 2))






'''Time 1'''

t1_interval = [0, 180]
t1 = np.arange(min(t1_interval), max(t1_interval))
t1p = np.linspace(min(t1), max(t1), 10*len(t1))
cases1 = np.array(cases)[min(t1):max(t1)+1]

#set parameter guesses
n = 25.69e6
i0 = 150
beta1 = 2e-9
gamma1 = 1/15.2
r0 = 1000

params1 = [beta1, n-i0, i0, r0] #beta, S(0), I(0), R(0)

#test other guesses

p1 = [3e-7, n-i0-r0,  i0, r0]
p2 = [1e-5, n-i0-10000, i0, 10000]

sse = [get_sse4(params1,t1), get_sse4(p1, t1), get_sse4(p2, t1)]
### Print outputs of sse for changes in parameters (using f' strings again like in assignment 2 code)
print(f'''
p0 sse = {get_sse4(params1, t1)}
p1 sse = {get_sse4(p1,t1)}
p2 sse = {get_sse4(p2,t1)}
min sse = {min(sse)}''')

'''this function optimizes params s.t. SSE is minimized 
(I gave it 10000 iterations to try for convergence)'''

soln1 = opt.fmin(get_sse4, p1, args=(t1,), maxiter = 10000) 
print(soln1)
best_result1 = solve4(soln1, t1p) #solve system with optimized paramaters

#plot first time interval fit
plt.plot(t1,cases1, 'bo', label = 'Observed Active Cases', )
plt.plot(t1p, best_result1[:,1], 'r', linewidth= 4, label = 'Fitted Model')
plt.title('4 Parameter Model of the First Wave',fontsize = 15, fontweight= 'bold', pad=10)
plt.text(75, 15000, r'$\beta$ = $1.22 x 10^{-6}$', fontsize = 15, fontweight='bold')
plt.text(75, 12000, r'$\gamma$ = 0.0658', fontsize=15)
plt.legend(loc='upper left', prop=dict(weight='bold'))
plt.xlabel('Time (days)', fontweight='bold', fontsize = 15, labelpad = 10 )
plt.ylabel('Active Cases', fontweight = 'bold', fontsize = 15, labelpad = 10)
plt.xticks(fontweight='bold', fontsize = 13)
plt.yticks(fontweight='bold', fontsize = 13)
plt.show()


#SSE and AIC 
sse41= get_sse4(soln1, t1)
aic41 = len(t1)*np.log(sse41/len(t1))+2*len(soln1)


'''Time 2'''

t2_interval = [180, 213]
t2 = np.arange(min(t2_interval), max(t2_interval))
t2p = np.linspace(min(t2), max(t2), 10*len(t2))
cases2 = np.array(cases)[min(t2):max(t2)+1]

#set parameter guesses
n = 25.69e6
i1 = cases1[max(t1)]
beta2 = 2e-9
gamma2 = 1/15.2
mu2 = 0.006
r1 = best_result1[:,2][max(t1)]
y1 = [n-i0, i1, 0]
params2 = [beta2,gamma2,n-i1-100, i1, 100]


# Solve ode(s)
result2 = solve4(params2,t2p)
labels = ['T(t)','I(t)']



#test other guesses

p1 = [3e-10, best_result1[:,0][max(t1)], i1, r1]
p2 = [1e-11,  n-i1-r1, i1, r1]

sse = [get_sse4(params2,t2), get_sse4(p1, t2), get_sse4(p2, t2)]
### Print outputs of sse for changes in parameters (using f' strings again like in assignment 2 code)
print(f'''
p0 sse = {get_sse4(params2, t2)}
p1 sse = {get_sse4(p1,t2)}
p2 sse = {get_sse4(p2,t2)}
min sse = {min(sse)}''')

soln2 = opt.fmin(get_sse4, p2, args=(t2,), maxiter = 10000)
print(soln2)
best_result2 = solve4(soln2, t2p)


plt.plot(t2,cases2, 'bo', label = 'Observed Active Cases')
plt.plot(t2p, best_result2[:,1], 'r', linewidth=4, label = 'Fitted Model')
plt.title('4 Parameter Model of the Second Wave', fontsize = 15, fontweight= 'bold', pad=10)
plt.text(195, 28000, r'$\beta$ = $7.16 x 10^{-7}$', fontsize = 15, fontweight='bold')
plt.text(195, 26500, r'$\gamma$ = 0.0658', fontsize=15)
plt.legend(prop=dict(weight='bold'))
plt.xlabel('Time (days)', fontweight='bold', fontsize = 15, labelpad = 10 )
plt.ylabel('Active Cases', fontweight = 'bold', fontsize = 15, labelpad = 10)
plt.xticks(fontweight='bold', fontsize = 13)
plt.yticks(fontweight='bold', fontsize = 13)
plt.show()

'''AIC'''
sse42= get_sse4(soln2, t2)
aic42 = len(t2)*np.log(sse42/len(t2)) + 2*len(soln2)


'''5-Parameter fit'''


''' Set up Functions '''
# Set up ode solver with 'inline' function definition

def solve5(p,time):
    beta = p[0]
    gamma = p[1]
    s = p[2]
    i = p[3]
    r = p[4]
    sys = lambda u,t: [-1*beta*u[0]*u[1], beta*u[0]*u[1] - gamma*u[1], gamma*u[1]]
    result = odeint(sys, [s,i,r], time)
    return(result)





def get_sse2(p,time):
    tp = np.linspace(min(time), max(time), 10*len(time))
    obs = np.array(cases)[min(time):max(time)+1]
    result = solve5(p, tp)
    f = inter.interp1d(tp, result[:, 1])
    i_new = f(time)
    resid = obs - i_new
    return (sum(resid ** 2))




'''Time 1'''

t1_interval = [0, 180]
t1 = np.arange(min(t1_interval), max(t1_interval))
t1p = np.linspace(min(t1), max(t1), 10*len(t1))
cases1 = np.array(cases)[min(t1):max(t1)+1]

#set parameter guesses
n = 25.69e6
i0 = 150
beta1 = 2e-9
gamma1 = 1/15.2
mu1 = 0.006
r0 = 1000
y0 = [n-i0-1000, i0, 1000]
params1 = [beta1,gamma1, n-i0, i0, r0]


#test other guesses

p1 = [3e-9, 1/70, n-i0, i0, 0]
p2 = [3e-8, 1/15, n-i0, i0, 0]

sse = [get_sse2(params1,t1), get_sse2(p1, t1), get_sse2(p2, t1)]
### Print outputs of sse for changes in parameters (using f' strings)
print(f'''
p0 sse = {get_sse2(params1, t1)}
p1 sse = {get_sse2(p1,t1)}
p2 sse = {get_sse2(p2,t1)}
min sse = {min(sse)}''')

soln1 = opt.fmin(get_sse2, p2, args=(t1,), maxiter = 10000)
print(soln1)
best_result1 = solve5(soln1, t1p)
plt.plot(t1,cases1, 'b')
plt.plot(t1p, best_result1[:,1], 'r')
plt.title('Soln to t1')
plt.show()
sse51 = get_sse2(soln1, t1)
aic51 = len(t1)*np.log(sse51/len(t1))+2*6

'''Time 2'''

t2_interval = [180, 213]
t2 = np.arange(min(t2_interval), max(t2_interval))
t2p = np.linspace(min(t2), max(t2), 10*len(t2))
cases2 = np.array(cases)[min(t2):max(t2)+1]

#set parameter guesses
n = 25.69e6
i1 = cases1[max(t1)]
beta2 = 2e-9
gamma2 = 1/15.2
mu2 = 0.006
r1 = best_result1[:,2][max(t1)]
y1 = [n-i0, i1, 0]


# Solve ode(s)
params2 = [beta2,gamma2, soln1[4], i1, r1]
result2 = solve5(params2,t2p)
labels = ['T(t)','I(t)']

### Show initial plot of obs data and inital model:

plt.xlabel('Time (days)', weight='bold')
plt.title('Solution to Model with Initial Guesses', weight='bold')
plt.ylabel('# of People', weight='bold')



plt.plot(t2,cases2, 'b')
plt.plot(t2p, result2[:,1], 'r')
# plt.plot(t, model)
plt.show()



#test other guesses

p1 = [4e-8, 1/25,n-i1-r1, i1, r1]
p2 = [1e-6, 1/50, n-i1-r1, i1, sum(cases1)]

sse = [get_sse2(params2,t2), get_sse2(p1, t2), get_sse2(p2, t2)]
### Print outputs of sse for changes in parameters (using f' strings again like in assignment 2 code)
print(f'''
p0 sse = {get_sse2(params2, t2)}
p1 sse = {get_sse2(p1,t2)}
p2 sse = {get_sse2(p2,t2)}
min sse = {min(sse)}''')

soln2 = opt.fmin(get_sse2, p1, args=(t2,), maxiter = 10000)
print(soln2)
best_result2 = solve5(soln2, t2p)
plt.plot(t2,cases2, 'b')
plt.plot(t2p, best_result2[:,1], 'r')
plt.title('Soln to t2')
plt.show()

sse52= get_sse2(soln2, t2)
aic52 = len(t2)*np.log(sse52/len(t2)) + 2*len(soln2)




plt.plot(t, cases,'ko', label = 'Observed Data')
plt.title('Fit of 5-Parameter Model', fontweight='bold', fontsize = 15, pad=10)
plt.xlabel('Time (days)', fontweight='bold', fontsize= 15, labelpad = 10)
plt.ylabel('Active Cases', fontweight='bold', fontsize = 15, labelpad=10)
plt.plot(t1p, best_result1[:,1], 'c',linewidth=4, label = 'Fitted Model for Time 1')
plt.plot(t2p, best_result2[:,1],'r', linewidth =4,label = 'Fitted Model for Time 2' )
#plt.plot(t3p, best_result3[:,1],'orange', linewidth =3 , label = 'Fitted Model for Time 3')
#plt.plot(t4p, best_result4[:,1],'magenta', linewidth =3 , label = 'Fitted Model for Time 4')
plt.xticks(fontweight='bold', fontsize = 13)
plt.yticks(fontweight='bold', fontsize = 13)
plt.legend()
plt.show()











'''Plot the fits vs. Active Cases'''

plt.plot(t, cases,'ko', label = 'Observed Data')
plt.title('Fit of 4-Parameter Model', fontweight='bold', fontsize = 15)
plt.xlabel('Time (days)', fontweight='bold', fontsize= 15, labelpad=15)
plt.ylabel('Active Cases', fontweight='bold', fontsize = 15, labelpad = 15)
plt.plot(t1p, best_result1[:,1], 'c',linewidth=4, label = 'Fitted Model for Time 1')
plt.plot(t2p, best_result2[:,1],'r', linewidth =4,label = 'Fitted Model for Time 2')
plt.legend(fontsize=13)
plt.xticks(fontweight='bold', fontsize = 13)
plt.yticks(fontweight='bold', fontsize = 13)
plt.show()



'''AIC PLOTS and REL PROB'''


plt.plot([2, 4, 5], [aic21, aic41, aic51], 'o')
plt.title('AIC for Time Interval 1 Models', fontweight = 'bold', fontsize=15)
plt.xlabel('# of Parameters', fontweight='bold', labelpad=10, fontsize=15)
plt.ylabel('AIC', fontweight='bold', labelpad = 10, fontsize=15)
plt.xticks([2,3,4,5],fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()



plt.plot([2, 4, 5], [aic22, aic42, aic52], 'o')
plt.title('AIC for Time Interval 2 Models', fontweight = 'bold', fontsize=15)
plt.xlabel('# of Parameters', fontweight='bold', labelpad=10, fontsize=15)
plt.ylabel('AIC', fontweight='bold', labelpad = 10, fontsize=15)
plt.xticks([2,3,4,5],fontweight='bold')
plt.yticks([390, 395, 400, 405],fontweight='bold')
# =============================================================================

# =============================================================================
plt.show()

aic2 = [aic22, aic42, aic52]
aic1 = [aic21, aic41, aic51]
relprob1 = [np.exp(-1*(aic21-min(aic1))/2), np.exp(-1*(aic41-min(aic1))/2), np.exp(-1*(aic51-min(aic1))/2)]
relprob2 = [np.exp(-1*(aic22-min(aic2))/2), np.exp(-1*(aic42-min(aic2))/2), np.exp(-1*(aic52-min(aic2))/2)]



