#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


data = np.load("modeltrain.npz")
W = data["arr_0"]
b = data["arr_1"]


# In[3]:


def inference_nb(X, w, b):
    scores = X @ w + b
    labels = (scores > 0).astype(int)
    return labels


# In[4]:


data =  np.loadtxt("test.txt.gz")
X = data[:, :-1]
Y = data[:, -1]

predictions = inference_nb(X, W, b)
accuracy = (predictions == Y).mean()
print("Test accuracy: ", accuracy * 100)


# In[ ]:





# In[ ]:




