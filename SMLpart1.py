#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("train.csv", sep = ";", decimal = ",")
df = df.iloc[:, :14]
df.head(2)


# In[3]:


df.Lead.value_counts().plot(kind='pie')
plt.savefig('Q1', bbox_inches='tight')


# In[4]:


df['totalemale']=0
df['totalfemale']=0


# In[5]:


df.head()


# In[6]:


index = df.Lead == 'Female'
index0 = df.Lead == 'male'


# In[7]:


df.loc[index, 'totalfemale']  = \
    df.loc[index, 'Number words female'] +\
         df.loc[index, 'Number of words lead'] 
df.loc[~index, 'totalfemale']  =\
     df.loc[~index, 'Number words female']
df.loc[index0, 'totalmale']  = \
    df.loc[index0, 'Number words male'] +\
     df.loc[index0, 'Number of words lead'] 
df.loc[~index0, 'totalmale'] = df.loc[~index0, 'Number words male']


# In[8]:


df.head()


# In[10]:


dg = df.groupby(['Year'])[['totalmale', 'totalfemale']].\
    agg(['sum'])   


# In[13]:


#dg.head()


# In[12]:
  
dg.plot(figsize=(20,10), kind = "bar") 
plt.savefig('Q2', bbox_inches='tight')


# In[19]:


df["maleMore"] = df["totalmale"] - df["totalfemale"]


# In[21]:


df[df["maleMore"]>=0]["Gross"].sum(), \
    df[df["maleMore"]<0]["Gross"].sum()


# In[ ]:





# In[ ]:





# In[ ]:




