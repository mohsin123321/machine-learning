#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import os
from aclImdb.porter import stem


# In[2]:


def load_vocabulary(filename):
    f = open(filename, encoding="utf8")
    text = f.read()
    words = text.split()
    voc = {}
    index = 0
    for word in words:
        voc[word] = index
        index += 1
    f.close()
    return voc


# In[3]:


PUNCT =  "!#$%&()'*+-/.,:;@?[]{}|^_`~<>=\"\\"
TABLE = str.maketrans(PUNCT," " * len(PUNCT))


# In[4]:


def read_document(filename,voc):
    f = open(filename, encoding="utf8")
    text = f.read()
    f.close()
    text = text.lower()
    text = text.translate(TABLE)
    words = text.split()
    bow = np.zeros(len(voc))
    for word in words:
        #word = stem(word)
        if word in voc:
            index = voc[word]
            bow[index] += 1
    return bow


# In[5]:


vocabulary = load_vocabulary("vocabulary.txt")


# In[6]:


documents = []
labels = []
for f in os.listdir("aclImdb/train/pos"):
    path = "aclImdb/train/pos/" + f
    bow = read_document(path, vocabulary)
    documents.append(bow)
    labels.append(1)
for f in os.listdir("aclImdb/train/neg"):
    path = "aclImdb/train/neg/" + f
    bow = read_document(path, vocabulary)
    documents.append(bow)
    labels.append(0)

X = np.stack(documents)
Y = np.array(labels)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("train.txt.gz", data)


# In[7]:


documents = []
labels = []
for f in os.listdir("aclImdb/test/pos"):
    path = "aclImdb/test/pos/" + f
    bow = read_document(path, vocabulary)
    documents.append(bow)
    labels.append(1)
for f in os.listdir("aclImdb/test/neg"):
    path = "aclImdb/test/neg/" + f
    bow = read_document(path, vocabulary)
    documents.append(bow)
    labels.append(0)

X = np.stack(documents)
Y = np.array(labels)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("test.txt.gz", data)


# In[8]:


documents = []
labels = []
for f in os.listdir("aclImdb/validation/pos"):
    path = "aclImdb/validation/pos/" + f
    bow = read_document(path, vocabulary)
    documents.append(bow)
    labels.append(1)
for f in os.listdir("aclImdb/validation/neg"):
    path = "aclImdb/validation/neg/" + f
    bow = read_document(path, vocabulary)
    documents.append(bow)
    labels.append(0)

X = np.stack(documents)
Y = np.array(labels)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("validation.txt.gz", data)


# In[ ]:




