#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


df=pd.read_csv("Melbourne_housing_FULL.csv")
df.nunique()


# In[3]:


df.drop(["Suburb", "Type", "Date", "Postcode","YearBuilt","Lattitude", "Longtitude"], axis=1, inplace=True)

df = pd.get_dummies(df, dtype='int')


# In[4]:


col=['Suburb','Rooms','Type','Method','SellerG','Regionname','Propertycount','Distance','CouncilArea','Bedroom2','Bathroom','Car','Landsize','BuildingArea']


# In[5]:


df.dropna(inplace=True)
df


# In[6]:


X=df.drop(columns=['Price'])
Y=df['Price']
X
Y


# In[7]:


X_train_lin, X_test_lin, Y_train_lin, Y_test_lin = train_test_split(X, Y, test_size=0.2, random_state=42)
lin_model = LinearRegression()
lin_model.fit(X_train_lin, Y_train_lin)
lin_model.score(X_train_lin,Y_train_lin)


# In[11]:


lin_model.score(X_test_lin,Y_test_lin)


# In[13]:


from sklearn.linear_model import Lasso
reg2=Lasso(alpha=50,max_iter=100,tol=0.1)
reg2.fit(X_train_lin,Y_train_lin)


# In[14]:


reg2.score(X_train_lin,Y_train_lin)


# In[15]:


reg2.score(X_test_lin,Y_test_lin)


# In[16]:


from sklearn.linear_model import Ridge
import numpy as np 
reg3=Ridge(alpha=50,max_iter=100,tol=0.1)
reg3.fit(X_train_lin,Y_train_lin)


# In[17]:


reg3.score(X_train_lin,Y_train_lin)


# In[18]:


reg3.score(X_train_lin,Y_train_lin)


# In[ ]:




