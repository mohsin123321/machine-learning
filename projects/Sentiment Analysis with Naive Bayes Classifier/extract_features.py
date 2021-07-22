
import numpy as np 
import os
from aclImdb.porter import stem

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

PUNCT =  "!#$%&()'*+-/.,:;@?[]{}|^_`~<>=\"\\"
TABLE = str.maketrans(PUNCT," " * len(PUNCT))

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

vocabulary = load_vocabulary("vocabulary.txt")

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




