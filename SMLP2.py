#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl_lm
import sklearn.preprocessing as skl_pre
from sklearn.metrics import confusion_matrix as cm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from statistics import mean


# In[2]:


dT = pd.read_csv("test.csv", sep = ",", decimal = ",")
dT = dT.iloc[:, :15]
dT.head(2)


# In[3]:


df = pd.read_csv("train.csv", sep = ";", decimal = ",")
df = df.iloc[:, :14]
df.head(2)


# In[4]:


np.random.seed(2)
index = np.random.choice(len(df), int(0.8* len(df)),\
replace = False)
index = df.index.isin(index)
train = df[index]
test = df[~index]


# In[27]:


Xtrain=train.iloc[:, :13]
ytrain = train.iloc[:, 13]

Xtest=test.iloc[:, :13]
ytest = test.iloc[:, 13]


# In[28]:


pd.plotting.scatter_matrix(Xtrain, figsize=(15,15))
plt.savefig('scatter', bbox_inches='tight')
plt.show()


# In[29]:


sc = skl_pre.StandardScaler()
le = skl_pre.LabelEncoder()


# In[8]:


Xs = sc.fit_transform(Xtrain)
ys = le.fit_transform(ytrain)
Xt = sc.transform(Xtest)
yt = le.transform(ytest)


# In[9]:


Xs = pd.DataFrame(sc.fit_transform(Xtrain), \
    columns = Xtrain.columns)
ys = ytrain
Xt = pd.DataFrame(sc.transform(Xtest), \
    columns = Xtrain.columns) 
yt = ytest


# In[10]:


model = skl_lm.LogisticRegression(solver="lbfgs")
model.fit(Xs, ys)


# In[11]:


prob = model.predict_proba(Xt)
print(model.classes_)
prob[:5]


# In[12]:


predict = np.empty(len(Xt), dtype= object)
predict = np.where(prob[:, 0] >= 0.5, 'Female', 'Male')
predict[:5]


# In[13]:


score = model.predict(Xt)


# In[14]:


pd.crosstab(score, yt)


# In[15]:


print ('Confusion Matrix')
cm(yt, score)


# In[16]:


acc = np.mean(yt ==score)
acc


# ## Cross Validation
# 

# In[17]:


scores = cross_val_score(model, Xs, ys, cv=5)


# In[18]:


scores


# In[19]:


np.mean(scores)


# In[20]:


np.sum(ys == 'Male'), np.sum(ys == 'Female')


# ## Stratified K Fold Cross Validation
# 

# In[21]:


#Reference: https://www.geeksforgeeks.org/ \\
# stratified-k-fold-cross-validation/


# In[22]:


skf = StratifiedKFold(n_splits=5)
scores = []
for train_index, test_index in skf.split(Xs, ys):
    X_train, X_test = Xs.iloc[train_index, :],\
         Xs.iloc[test_index, :]
    y_train, y_test = ys.iloc[train_index],\
        ys.iloc[test_index]
    model = skl_lm.LogisticRegression(solver="lbfgs")
    model.fit(X_train, y_train)
    scores.append(accuracy_score(model.predict(X_test),\
         y_test))
scores


# In[23]:


np.mean(scores)


# In[24]:


class NavClassifier:
    def __init__(self):
        pass   
    def predict(self, X):
        return pd.DataFrame({"Lead":['Male']*len(X)})


# In[25]:


naive = NavClassifier()


# In[26]:


accuracy_score(naive.predict(Xs), ys)

