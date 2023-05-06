#!/usr/bin/env python
# coding: utf-8

# # predicting the house price based on the information of houses in tehran province.

# ### in this project I have used two models.
# ### one model is a non-linear regression based on one variable and the other one is a multiple linear regression based on three variables.
# ### in both models the maximum reachable R2-score was 0.72

# In[129]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[163]:


df=pd.read_csv(r"C:\Users\mohad\Downloads\1632300362534233.csv")
df.head(10)


# In[44]:


df.describe()


# ### removing null addresses

# In[164]:


df['Address'].dropna()


# ### removing big areas

# In[165]:


df['Area'] = df['Area'].str.replace(',', '')
df['Area'] = df['Area'].astype(float)
df = df[df['Area'] < 1000]
print(len(df.Area))


# ### relation between area and price is similar to a logarithmic relation

# In[158]:


plt.scatter(df.Area,df.Price,color='blue')
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()


# ### converting addresses into numerical values
# #### label encoder assigns each address a value alphabetically which is not desired and I think thats why 
# #### the chart does not give an comprehensible vision to figure out the relation.

# In[185]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df.Address = label.fit_transform(labeledAddress.Address)
print(df.Address)
plt.scatter(df.Address,df.Price,color='blue')
plt.xlabel('Address')
plt.ylabel('Price')
plt.show()


# ### creating a dataframe which indicates each region and their price per area

# In[99]:


pricePerArea=df['Price']/df['Area']
d = {'Address': df['Address'], 'pricePerArea': pricePerArea}
importances=pd.DataFrame(data=d)
print(importances)


# ### sorting addresses according to pricePerAea in ascending order.which means regions in downtown(lower price per area) get lower order.

# In[101]:


from sklearn.linear_model import LogisticRegression
labeledAddress = importances.sort_values(by='pricePerArea')
print(labeledAddress)


# ### we can assume an approximate line. although the relation between address and price is somehow non-linear and logarithmic

# In[107]:


import seaborn as sns
sns.catplot(data=labeledAddress, x="Address", y="pricePerArea")


# ### the relation between room and price is somehow logarithmic

# In[10]:


plt.scatter(df.Room,df.Price,color='blue')
plt.xlabel("Room")
plt.ylabel("Price")
plt.show()


# In[12]:


plt.scatter(df.Parking,df.Price,color='blue')
plt.xlabel("Parking")
plt.ylabel("Price")
plt.show()


# ### comparing three features(address,area,room) together

# In[348]:


plt.scatter(df.Address,df.Price,color='blue')
plt.scatter(df.Room,df.Price,color='red')
plt.scatter(df.Area,df.Price,color='green')


# In[13]:


plt.scatter(df.Warehouse,df.Price,color='blue')
plt.xlabel("Warehouse")
plt.ylabel("Price")
plt.show()


# In[143]:


plt.scatter(df.Elevator,df.Price,color='blue')
plt.xlabel("Elevator")
plt.ylabel("Price")
plt.show()


# ### selecting required columns
# #### area,address and room are features that influence the price the most according to charts

# In[186]:


newdf=df[['Area','Address','Room','Price']]
newdf.head()


# ### normalizing Area and Price features

# In[205]:


newdf['Area']=newdf['Area']/max(newdf['Area'])
newdf['Price']=newdf['Price']/max(newdf['Price'])
print(newdf['Area'])


# ### creating train/test datasets

# In[344]:


msk=np.random.rand(len(df))<0.9
train=newdf[msk]
test=newdf[~msk]
fig=plt.figure()
plot1=fig.add_subplot()
plot1.scatter(train['Area'],train['Price'],color='blue')
plot1.scatter(test['Area'],test['Price'],color='red')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()


# ### first non-linear model based on Area feature.

# In[181]:


def logarithm(x,a,b):
      return np.log(a*x+b)


# In[345]:


from scipy.optimize import curve_fit
popt, pcov = curve_fit(logarithm,train['Area'],train['Price'])
print(popt)
print(pcov)
#print the final parameters
print(" a = %f, b = %f" % (popt[0], popt[1]))


# In[346]:


y_hat=logarithm(test['Area'],*popt)
plt.scatter(test['Area'],test['Price'],color='blue')
plt.scatter(test['Area'],y_hat,color='red')
plt.show()


# In[347]:


from sklearn.metrics import r2_score
print("MSE is : %d",np.mean(((y_hat-test['Price'])**2)))
print("R2-score: %.2f" % r2_score(test['Price'],y_hat ))


# ### second linear model based on Area,Address and Room features.

# In[349]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['Area','Address','Room']])
y = np.asanyarray(train[['Price']])
regr.fit (x, y)
print ('Coefficients: ', regr.coef_)


# In[352]:


y_hat1= regr.predict(test[['Area','Address','Room']])
x = np.asanyarray(test[['Area','Address','Room']])
y = np.asanyarray(test[['Price']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat1 - y) ** 2))

# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % regr.score(x, y))
print("R2-score: %.2f" % r2_score(test['Price'],y_hat1))


# In[ ]:




