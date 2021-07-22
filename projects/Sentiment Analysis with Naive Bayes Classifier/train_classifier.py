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


#Top 10 impactful words for the classification of model
f = open("vocabulary.txt", encoding="utf8")
vocabulary = f.read().split()
f.close()
indices = np.argsort(w)
print("NEGATIVE WORDS")
for i in indices[:10]:
	print(vocabulary[i], w[i])
	print("POSITIVE WORDS")
for i in indices[-10:]:
	print(vocabulary[i], w[i])

#Top 10 missclassified documents
documents = []
for f in os.listdir("aclImdb/smalltrain/pos"):
	path = f
	documents.append(path)
for f in os.listdir("aclImdb/smalltrain/neg"):
	path = f
	documents.append(path)


#extracting missclassified documents
mis_clsfd_doc = [(idx,v) for idx,v in enumerate(scores) if Y[idx] != (scores[idx] > 0).astype(int)]
np_mis_clsfd = np.empty( len(mis_clsfd_doc),dtype=[('index', int), ('value', float)])
np_mis_clsfd[:] = mis_clsfd_doc
np_mis_clsfd = np.sort(np_mis_clsfd, order='value') 

print('Negative Missclassified reviews')
# top 10 negative missclassified documents
for tuple in np_mis_clsfd[:10]:
	path = "aclImdb/smalltrain/pos/"
	f = open(path + documents[tuple[0]], encoding="utf8")
	text = f.read()
	f.close()
	print(text)
	print('-------------------------------------------------------------')

print('Positive Missclassified reviews')
# top 10 positive missclassified documents
for tuple in np_mis_clsfd[-10:]:
	path = "aclImdb/smalltrain/neg/"
	f = open(path + documents[tuple[0]], encoding="utf8")
	text = f.read()
	f.close()
	print(text)
	print('-------------------------------------------------------------')








