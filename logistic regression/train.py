#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

def load_file(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1] # data[:, data.shape[1] -1]
    Y = data[:, -1] # data[:, data.shape[1]]
    return X, Y

X, Y = load_file('titanic-train.txt')
print ("Loaded", X.shape[0], "feature Vectors")


# In[2]:


def logreg_inference(X,W,b):
    z = ( X @ W ) + b
    p = 1/(1 + np.exp(-z))
    return p


# In[3]:


def logreg_train(X , Y , lambda_ , lr =0.001 , steps =100):
    m, n = X.shape
    b = 0
    W = np.zeros(n)
    acc = []
    losses = []
    for step in range(steps):
        P = logreg_inference(X, W, b) 
        if (step % 1000 == 0 ):
            loss = cross_entropy(P,Y)
            predictions = (P > 0.5)
            accuracy = ( predictions == Y ).mean()
            acc.append(accuracy * 100)
            losses.append(loss * 100)
        grad_b = (P - Y).mean() # ((p-y)/m)
        grad_w = ( (X.T @ (P - Y)) / m ) 
        
        b -= lr * grad_b
        W -= lr * grad_w
        
    return W, b , acc, losses


# In[4]:


def cross_entropy(P, Y):
    #epsilon = 1e-5    
    return (-Y * np.log(P) - (1 - Y) * np.log(1-P)).mean()


# In[5]:


import pandas as pd
#df_Y = pd.DataFrame(Y,columns=['pred'])
df = pd.DataFrame(X,columns=['class','sex','age','spouses','parents','fare'])
#df = pd.concat([df_X,df_Y],axis=1)
#df 


# In[6]:


df['class'] = np.log10(df['class'] + (df.loc[ df['class'] != 0].min())['class'])
df['sex'] = np.log10(df.sex + df.loc[ df['sex'] != 0].min().sex)
df['age'] = np.log10(df.age + df.loc[ df['age'] != 0].min().age)
df['spouses'] = np.log10(df.spouses + df.loc[ df['spouses'] != 0].min().spouses)
df['parents'] = np.log10(df.parents + df.loc[ df['parents'] != 0].min().parents)
df['fare'] = np.log10(df.fare + df.loc[ df['fare'] != 0].min().fare)


# In[7]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_stand = pd.DataFrame(scaler.fit_transform(df), index = df.index , columns = df.columns)
df_stand.describe()


# In[8]:


df['parents'].hist()


# In[ ]:





# In[9]:


corr = df_stand.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[10]:


temp_X = df_stand.to_numpy()
temp_X


# In[11]:


X = temp_X+
X


# In[12]:


W, b, acc,losses = logreg_train(X, Y, 0, 0.005, 1000000)
print("W =", W)
print("b = ", b)


# In[13]:


plt.title("Accuracy (%)")
plt.plot(acc)
plt.savefig('accuracy.jpeg')
plt.show()


# In[14]:


plt.title("Loss (%)")
plt.plot(losses)
plt.savefig('Loss.jpeg')
plt.show()


# In[15]:


X_ = np.array([[1, 0, 24, 0, 0, 35.1]])
P = logreg_inference(X_, W, b)
print(P)


# In[16]:


P = logreg_inference(X, W, b)
predictions = (P > 0.5)
accuracy = ( predictions == Y ).mean()
print("Accuracy =",accuracy * 100)


# In[17]:


Xrnd = X + np.random.randn(X.shape[0], X.shape[1])/12
plt.scatter(Xrnd[:,0],Xrnd[:,1], c=Y)
plt.xlabel('Class')
plt.ylabel('Gender')
plt.colorbar()
plt.savefig('Distribution.jpeg')
plt.show()


# In[18]:


np.savez("model.npy",W,b)

