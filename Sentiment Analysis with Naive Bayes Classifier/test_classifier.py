import numpy as np

data = np.load("modeltrain.npz")
W = data["arr_0"]
b = data["arr_1"]

def inference_nb(X, w, b):
    scores = X @ w + b
    labels = (scores > 0).astype(int)
    return labels

data =  np.loadtxt("test.txt.gz")
X = data[:, :-1]
Y = data[:, -1]

predictions = inference_nb(X, W, b)
accuracy = (predictions == Y).mean()
print("Test accuracy: ", accuracy * 100)
