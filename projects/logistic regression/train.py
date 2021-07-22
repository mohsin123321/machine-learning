import numpy as np
import matplotlib.pyplot as plt

def load_file(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1] # data[:, data.shape[1] -1]
    Y = data[:, -1] # data[:, data.shape[1]]
    return X, Y

X, Y = load_file('titanic-train.txt')
print ("Loaded", X.shape[0], "feature Vectors")


def logreg_inference(X,W,b):
    z = ( X @ W ) + b
    p = 1/(1 + np.exp(-z))
    return p


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


def cross_entropy(P, Y):
    #epsilon = 1e-5    
    return (-Y * np.log(P) - (1 - Y) * np.log(1-P)).mean()


W, b, acc,losses = logreg_train(X, Y, 0, 0.005, 1000000)
print("W =", W)
print("b = ", b)



plt.title("Accuracy (%)")
plt.plot(acc)
plt.savefig('accuracy.jpeg')
plt.show()


plt.title("Loss (%)")
plt.plot(losses)
plt.savefig('Loss.jpeg')
plt.show()



X_ = np.array([[1, 0, 24, 0, 0, 35.1]])
P = logreg_inference(X_, W, b)
print(P)



P = logreg_inference(X, W, b)
predictions = (P > 0.5)
accuracy = ( predictions == Y ).mean()
print("Accuracy =",accuracy * 100)


Xrnd = X + np.random.randn(X.shape[0], X.shape[1])/12
plt.scatter(Xrnd[:,0],Xrnd[:,1], c=Y)
plt.xlabel('Class')
plt.ylabel('Gender')
plt.colorbar()
plt.savefig('Distribution.jpeg')
plt.show()


np.savez("model.npy",W,b)

