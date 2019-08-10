#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6/26/19
@author: hollyerickson

This program reads the TrackMan Baseball files.
Shows left vs right EV & SLG by pitch type.
"""
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

#%%


df = pd.DataFrame()

for fname in glob.glob("./trackman data/2018 Game Split Data/2018 Game Split Data/*.csv"): # This will loop through all CSV files within the folder at path
        current_csv = pd.read_csv(fname, header=0)
        current_rows = current_csv.loc[(current_csv['KorBB'] != 'Walk') & (current_csv['PlayResult']!= 'Sacrifice')] # Where ExitSpeed is not NaN
        df = df.append(current_rows)   

print(df.shape) #(95890, 75)


#%%
slug_rel = df[['ExitSpeed', 'PlayResult', 'BatterSide', 'Distance', 'PitcherThrows', 'HangTime', 'Angle']]
ll = slug_rel.loc[(slug_rel['PitcherThrows']== 'Right') & (slug_rel['BatterSide'] == 'Left')]
lr = slug_rel.loc[(slug_rel['PitcherThrows']== 'Right') & (slug_rel['BatterSide'] == 'Right')]

#%%
plt.figure(figsize=(18, 12))
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].hist(lr['Angle'], bins = 100)
axs[0].set_title("Right Batter Angle")  
axs[1].hist(ll['Angle'], bins = 100)
axs[1].set_title("Left Batter Angle")
#sns.distplot(a= lr['Distance'], bins = 200, kde=True).set_title('Histogram of Right Batter Distance')
#plt.ylim(0.000, 0.035)
plt.show()
#%%

for index, row in ll.iterrows():
    if row['PlayResult'] == 'Single':
        val = 1
    elif row['PlayResult'] == 'Double':
        val = 2
    elif row['PlayResult'] == 'Triple':
        val = 3
    elif row['PlayResult'] == 'HomeRun':
        val= 4
    else:
        val = np.NAN
    ll.at[index,'Result'] = val
    
for index, row in lr.iterrows():
    if row['PlayResult'] == 'Single':
        val = 1
    elif row['PlayResult'] == 'Double':
        val = 2
    elif row['PlayResult'] == 'Triple':
        val = 3
    elif row['PlayResult'] == 'HomeRun':
        val= 4
    else:
        val = np.NAN
    lr.at[index,'Result'] = val
    
#%%
plt.figure(figsize=(18, 12))
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].hist(lr['Result'], bins = 100)
axs[0].set_title("Right Batter Hit Result")  
axs[1].hist(ll['Result'], bins = 100)
axs[1].set_title("Left Batter Hit Result")
#sns.distplot(a= lr['Distance'], bins = 200, kde=True).set_title('Histogram of Right Batter Distance')
#plt.ylim(0.000, 0.035)
#plt.xlim(0, 130)

plt.show()




#%%
print(ll['ExitSpeed'].mean())
print(lr['ExitSpeed'].mean())
print(ll['Angle'].mean())
print(lr['Angle'].mean())
print(ll['Result'].mean())
print(lr['Result'].mean())

#%%
llround = ll.round(0)
lrround = lr.round(0)
print(llround[['ExitSpeed', 'Angle', 'Result']].mode(dropna=True, axis=0))
print(lrround[['ExitSpeed', 'Angle', 'Result']].mode(dropna=True, axis=0))

#%%
# Get the spread
print(ll['ExitSpeed'].var())
print(lr['ExitSpeed'].var())
print(ll['Angle'].var())
print(lr['Angle'].var())
print(ll['Result'].var())
print(lr['Result'].var())

#%%
# Plot PMF for Exit velocity

plt.figure(figsize=(100, 12))

# Get the PMF
llget = llround['ExitSpeed'].dropna().value_counts()
lrget = lrround['ExitSpeed'].dropna().value_counts()
llget = llget.reset_index()
lrget = lrget.reset_index()
llget['pmf'] = llget['ExitSpeed'] / llround['ExitSpeed'].count()
lrget['pmf1'] = lrget['ExitSpeed'] / lrround['ExitSpeed'].count()

fig, ax = plt.subplots()


width = 0.5  # the width of the bars

result = pd.merge(lrget, llget, on="index")
 
ax.bar(result['index'] - width/2, result['pmf'], width=width, facecolor='red', label = 'Left')
ax.bar(result['index'] + width/2, result['pmf1'], width=width, facecolor='blue', label = 'Right')
plt.legend(['Left', 'Right'])

#%%
# Get CDFs
num_bins = 50
ll_college_2018, ll_bin_edges_college_2018 = np.histogram(ll['ExitSpeed'].dropna(), bins=num_bins, normed=True)
cdf_ll = np.cumsum (ll_college_2018)
lr_college_2018, lr_bin_edges_college_2018 = np.histogram(lr['ExitSpeed'].dropna(), bins=num_bins, normed=True)
cdf_lr = np.cumsum (lr_college_2018)

#%%
# Plot CDFs
plt.figure(figsize=(18,13))
sns.set_style("darkgrid")
plt.plot(ll_bin_edges_college_2018[1:], cdf_ll/cdf_ll[-1], label='Left')
plt.plot(lr_bin_edges_college_2018[1:], cdf_lr/cdf_lr[-1], label='Right')
plt.title('Exit Velocity CDF')
plt.xlabel('Exit Velocity (MPH)')
plt.ylabel('CDF')
plt.legend(loc='lower right')

#%%
# Plot scatter plots
both = slug_rel.loc[(slug_rel['PitcherThrows']== 'Right')]
for index, row in df.iterrows():
    if row['BatterSide'] == 'Left':
        val = 1
    else:
        val = 0
    both.at[index,'Side'] = val
    
#%%
both.plot.scatter(x = 'Side', y = 'ExitSpeed')
plt.title('Left Batters')

#%%
for index, row in both.iterrows():
    if row['PlayResult'] == 'Single':
        val = 1
    elif row['PlayResult'] == 'Double':
        val = 2
    elif row['PlayResult'] == 'Triple':
        val = 3
    elif row['PlayResult'] == 'HomeRun':
        val= 4
    else:
        val = np.NAN
    both.at[index,'Result'] = val

#%%
both.corr()

sns.lmplot(x='Result', y='Side', data = both) 

#%%
from scipy.stats import ttest_ind_from_stats
mean1 = ll['ExitSpeed'].mean()
mean2 = lr['ExitSpeed'].mean()
std1 = ll['ExitSpeed'].std()
std2 = lr['ExitSpeed'].std()
n1 = ll['ExitSpeed'].count()
n2 = lr['ExitSpeed'].count()

tstat, pvalue = ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2)
print(pvalue)

#%%
mean1 = ll['Result'].mean()
mean2 = lr['Result'].mean()
std1 = ll['Result'].std()
std2 = lr['Result'].std()
n1 = ll['Result'].count()
n2 = lr['Result'].count()

tstat, pvalue = ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2)
print(pvalue)

#%%
both.to_csv('data.csv')