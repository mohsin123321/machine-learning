#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

data = np.load("model.npz")
W = data["arr_0"]
b = data["arr_1"]


# In[2]:


def load_file(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]
    Y = data[:, -1] 
    return X, Y


# In[3]:


def logreg_inference(X,W,b):
    z = ( X @ W ) + b
    p = 1/(1 + np.exp(-z))
    return p


# In[4]:


X, Y = load_file('titanic-test.txt')
P = logreg_inference(X, W, b)
predictions = (P > 0.5)
accuracy = ( predictions == Y ).mean()
print("Accuracy =",accuracy * 100)


# In[ ]:




