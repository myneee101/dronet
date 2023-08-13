#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas import read_csv
from pandas import set_option
from matplotlib import pyplot
import numpy
from pandas.plotting import scatter_matrix
filename = 'D:\ML & DM\Midterm1\heart_risk.csv'
raw_data = read_csv(filename)
print(raw_data.shape)
description = raw_data.describe()
print(description)
peek = raw_data.head(10)
print (peek)


# In[2]:


(raw_data[raw_data.values == 99999].index)


# In[3]:


import os
import numpy as np 
from pandas import read_csv
from scipy import stats
z = np.abs(stats.zscore(raw_data))
threshold = 3
print(np.where (z > 3))


# In[11]:


from matplotlib import pyplot
import numpy
# Histogram
raw_data.hist()
pyplot.show()
#Density Plots
raw_data.plot(kind='density', subplots=True, layout=(5,5), sharex=False)
pyplot.show()
# Box and Whisker plots
raw_data.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False)
pyplot.show()
# plot correlation matrix
correlations = raw_data.corr()
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
pyplot.show()
# Scatterplot Matrix
scatter_matrix(raw_data)
pyplot.show()


# In[12]:


raw_data[['Age','Sex','CP','Trestbps','Chol']] = raw_data[['Age','Sex','CP','Trestbps','Chol']].replace(99999,np.NaN)
print(raw_data)


# In[13]:


from matplotlib import pyplot
import numpy
# Histogram
raw_data.hist()
pyplot.show()
#Density Plots
raw_data.plot(kind='density', subplots=True, layout=(5,5), sharex=False)
pyplot.show()
# Box and Whisker plots
raw_data.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False)
pyplot.show()
# plot correlation matrix
correlations = raw_data.corr()
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
pyplot.show()
# Scatterplot Matrix
scatter_matrix(raw_data)
pyplot.show()


# In[14]:


from pandas import read_csv
from pandas import set_option
from matplotlib import pyplot
import numpy
from pandas.plotting import scatter_matrix
filename = 'D:\ML & DM\Midterm1\heart_risk.csv'
names = ['Age','Sex','CP','Trestbps','Chol']
raw_data = read_csv(filename, names = names)
print(raw_data.shape)
description = raw_data.describe()
print(description)
peek = raw_data.head(10)
print (peek)


# In[15]:


raw_data[['Age','Sex','CP','Trestbps','Chol']] = raw_data[['Age','Sex','CP','Trestbps','Chol']].replace(99999,np.NaN)
print(raw_data)


# In[17]:


from pandas import read_csv
from pandas import set_option
from matplotlib import pyplot
import numpy
from pandas.plotting import scatter_matrix
# Histogram
raw_data.hist()
pyplot.show()
#Density Plots
raw_data.plot(kind='density', subplots=True, layout=(5,5), sharex=False)
pyplot.show()
# Box and Whisker plots
raw_data.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False)
pyplot.show()
# plot correlation matrix
correlations = raw_data.corr()
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
pyplot.show()
# Scatterplot Matrix
scatter_matrix(raw_data)
pyplot.show()


# In[21]:


from pandas import read_csv
from pandas import set_option
from matplotlib import pyplot
import numpy
filename = 'D:\ML & DM\Midterm1\heart_risk.csv'
raw_data = read_csv(filename)
print(raw_data.shape)
description = raw_data.describe()
print(description)


# In[22]:


from matplotlib import pyplot
import numpy
# Histogram
raw_data.hist()
pyplot.show()
#Density Plots
raw_data.plot(kind='density', subplots=True, layout=(5,5), sharex=False)
pyplot.show()
# Box and Whisker plots
raw_data.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False)
pyplot.show()
# plot correlation matrix
correlations = raw_data.corr()
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
pyplot.show()
# Scatterplot Matrix
scatter_matrix(raw_data)
pyplot.show()


# In[23]:


raw_data[['Age','Sex','CP','Trestbps','Chol']] = raw_data[['Age','Sex','CP','Trestbps','Chol']].replace(99999,np.NaN)
print(raw_data)


# In[24]:


from matplotlib import pyplot
import numpy
# Histogram
raw_data.hist()
pyplot.show()
#Density Plots
raw_data.plot(kind='density', subplots=True, layout=(5,5), sharex=False)
pyplot.show()
# Box and Whisker plots
raw_data.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False)
pyplot.show()
# plot correlation matrix
correlations = raw_data.corr()
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
pyplot.show()
# Scatterplot Matrix
scatter_matrix(raw_data)
pyplot.show()


# In[38]:


# SVM after cleaning
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
filename = 'D:\ML & DM\Midterm1\heart_risk.csv'
names = ['Age','Sex','CP','Trestbps','Chol']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:5]
Y = array[:,5]
kfold = KFold(n_splits=10, random_state=7)
model = SVC()
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# In[32]:


# Normalization
from sklearn.preprocessing import Normalizer
from pandas import read_csv
from numpy import set_printoptions
filename = 'D:\ML & DM\Midterm1\heart_risk.csv'
names = ['Age','Sex','CP','Trestbps','Chol']
dataframefilename = read_csv(filename, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:5]
Y = array[:,5]
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(normalizedX[0:5,:])


# In[33]:


# Standardization
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from numpy import set_printoptions
filename = 'D:\ML & DM\Midterm1\heart_risk.csv'
names = ['Age','Sex','CP','Trestbps','Chol']
dataframe = read_csv(filename, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])


# In[37]:


# Chi squared classification
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# Load the data
filename = 'D:\ML & DM\Midterm1\heart_risk.csv'
names = ['Age','Sex','CP','Trestbps','Chol']
raw_data = read_csv(filename, names=names)
array = raw_data.values
X = array[:,0:5]
Y = array[:,5]
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])


# In[ ]:


from numpy import loadtxt
file = open(filename, 'rt')  
raw_data = loadtxt(file, delimiter=",") 
for j in range (0,15):
    for i in raw_data[:,j]:
        if (i == 0)or (i == 99999):
            sum = sum + 1  
    print("column ", j , " = ", sum)

