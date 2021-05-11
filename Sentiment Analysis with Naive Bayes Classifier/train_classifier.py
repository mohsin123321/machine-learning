import numpy as np


def train_nb(X,Y):
    m = X.shape[0]
    n =  X.shape[1]
    pos_counter = X[Y==1, :].sum(0)
    pi_pos = ( pos_counter + 1) / ( pos_counter.sum() + n )
    
    neg_counter = X[Y==0, :].sum(0)
    pi_neg = ( neg_counter + 1) / ( neg_counter.sum() + n )
    prior_pos = Y.sum() / m
    
    prior_neg = 1 - prior_pos
    w = np.log(pi_pos) - np.log(pi_neg)
    b =  np.log(prior_pos) - np.log(prior_neg)
    return w,b


def inference_nb(X, w, b):
    scores = X @ w + b
    labels = (scores > 0).astype(int)
    return labels,scores

data =  np.loadtxt("train.txt.gz")
X = data[:, :-1]
Y = data[:, -1]

print("data loaded")

w, b = train_nb(X, Y)
print("classifier trained")
predictions,scores = inference_nb(X, w, b)
accuracy = (predictions == Y).mean()
print("Training accuracy: ", accuracy * 100)


np.savez("modeltrain.npz",w,b)


