# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 18:51:44 2021

@author: Katrina Lawrence
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
from scipy import interpolate as inter
from scipy.integrate import odeint
from sklearn.metrics import r2_score as r2

#Set up
''' Read Data'''
df = pd.read_csv('datathon_data.csv')
cases = df["new_cases_14_days"][431:645]
cases_list = cases.tolist()

t = np.arange(0,len(cases) )
plt.plot(t, cases, label = 'Observed Data')
plt.title('Covid-19 Pandemic in Spain')
plt.xlabel('Time (days)')
plt.ylabel('Active Cases')
plt.show()
t_p = np.linspace(0, len(cases), len(cases)*10)


# EXPONENTIAL AREA 1
t_exp = t[130:160]
c_exp = cases[130:160]
'''Have to make a time interval w length of growth period starting at zero to 
make any meaningful prediction'''

t_pred = np.arange(0, len(c_exp)) #prediction time interval
plt.plot(t_pred, np.log(c_exp), 'co', label='Observed Data')

'''Exponential growth -> log(y) = kt + b is a linear fit'''
#fit using polyfit and polyval
m,b = np.polyfit(t_pred, np.log(c_exp),1)
fit = np.polyval([m,b], t_pred)
#Check if I picked a good interval w r^2 value
rsq= r2(np.log(c_exp), fit)

#plot the line w fitted value and annotations
plt.plot(t_pred,fit ,'orange', label='Fitted Model', linewidth=3)
plt.text(13, 8.5, f'Percent Growth: {round((np.exp(m)-1)*100,3)}%\n R$^2$: {round(rsq,4)}', fontweight = 'bold', fontsize=13)
plt.legend()
plt.title('Exponential Growth Period 1 ', fontweight = 'bold', fontsize=15, pad = 10)
plt.xlabel('Time (days)', fontweight = 'bold', fontsize=15, labelpad = 10,)
plt.ylabel('log(Active Cases)', fontweight = 'bold', fontsize=15, labelpad = 10,)
plt.xticks(fontweight='bold', fontsize = 13)
plt.yticks(fontweight='bold', fontsize = 13)
plt.show()

'''Make prediction based on linear model (NOTE: these predictions are on the 
log of active cases so we gotta take exp(predictions) to see the actual active
cases projection.)'''

#Take a point prediction 14 days after the first growth period (30 days long)
prediction1 = np.polyval([m,b], 44)

#get the extrapolated line over those 44 days 
prediction_line1 = np.polyval([m,b], np.arange(30,45))


'''Do the same thing but for the second area of growth'''
t_exp = t[175:195]
c_exp = cases[178:198]

#time starting at zero because we only care about time wrt this growth period
t_pred = np.arange(0, len(c_exp))


plt.plot(t_pred, np.log(c_exp), 'ro', label='Observed Data')

#make fit
m1,b1 = np.polyfit(t_pred, np.log(c_exp),1)
fit = np.polyval([m1,b1], t_pred)

rsq= r2(np.log(c_exp), fit) #fit good?

plt.plot(t_pred,fit , label='Fitted Model', linewidth=3)
plt.text(8.8, 10.10, f'Percent Growth: {round((np.exp(m1)-1)*100,3)}%\n R$^2$: {round(rsq,4)}', fontweight = 'bold', fontsize=13)
plt.legend()
plt.title('Exponential Growth Period 2 ', fontweight = 'bold', fontsize=15, pad = 10)
plt.xlabel('Time (days)', fontsize=15, labelpad = 10, fontweight = 'bold')
plt.ylabel('log(Active Cases)',fontsize=15, labelpad = 10, fontweight = 'bold')
plt.xticks(fontweight='bold', fontsize = 13)
plt.yticks(fontweight='bold', fontsize = 13)
plt.show()


'''Point and line prediction for 15 days after the growth
 (thats the last datapoint
 available)'''

#### these are the log(active cases) predictions
prediction2 = np.polyval([m1,b1],35 )
prediction_line2 = np.polyval([m1,b1], np.arange(20,35))




#Plot the point predictions, prediction curve, and observed values
#Cases (non-exponential)
plt.plot(t[100:129], cases_list[100:129], 'ko')
plt.plot(t[160:177], cases_list[160:177], 'ko')
plt.plot(t[181:213], cases_list[181:213], 'ko')
plt.plot(t[199:213], cases_list[199:213], 'ko')

#Exponential areas 
plt.plot(t[130:160], cases_list[130:160], 'co', label = 'Exponential Growth Area 1')
plt.plot(t[178:198], cases_list[178:198], 'ro', label= 'Exponential Growth Area 1')


# =============================================================================
# #Prediction 1
# plt.plot(174, np.exp(prediction1), 'co', label='Prediction 1', linewidth=4)
# plt.plot(t[160:175], np.exp(prediction_line1), 'c--', linewidth=3)
# plt.plot([174, 174], [np.exp(prediction1)-2000, cases_list[180]], 'k-', linewidth=3)
# 
# #Prediction2
# plt.plot([213, 213], [np.exp(prediction2)-1000, cases_list[213]], 'k-', linewidth=3)
# plt.plot(t[198:213], np.exp(prediction_line2), 'm--', linewidth=3)
# plt.plot(213, np.exp(prediction2), 'mo', label='Prediction 2', linewidth=4)
# 
# =============================================================================

#Title and Label format stuff
plt.title('Areas of Approximate Exponential Growth', fontweight = 'bold', fontsize=15, pad = 10)
plt.xlabel('Time (days)', fontweight = 'bold', fontsize=15, labelpad = 10)
plt.ylabel('Active Cases', fontweight = 'bold', fontsize=15, labelpad = 10)

plt.legend( prop=dict(weight='bold'))


plt.xticks(fontweight='bold', fontsize = 13)
plt.yticks(fontweight='bold', fontsize = 13)
plt.show()
