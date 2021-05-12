# Machine Learning Algorithms 
## 1) Logistic Regression
Logistic Regression implemented in Python from sratch. It uses gradient decent approach for the convergence of the model and uses the titanic data to calculate the probability of survivals.

## 2) Naive Bayes Algorithm
We used Naive Bayes classifier to predict the sentiments of the reviews, based on IMDB dataset , to predict them deemed negative or positive.
## 2.1) Data Preprocessing & Feature Extraction
Our dataset (attached in zip file __aclImdb.zip__) includes 50K reviews which are equally divided among positive and negative reviews from which 25K are for training, 12.5K are for testing and 12.5K are for validation. For making the computations and testing faster, we are considering small training data (approx. 6.2K for both positive and negative reviews) and later we will move to the original training data. Firstly, we build the vocabulary of words which counts 1K most common words which are used in the reviews both positive and negative.
```python   
for f in os.listdir("aclImdb/smalltrain/pos"):
path = "aclImdb/smalltrain/pos/" + f
words = read_document(path)
vocabulary.update(words)
for f in os.listdir("aclImdb/smalltrain/neg"):
path = "aclImdb/smalltrain/neg/" + f
words = read_document(path)
vocabulary.update(words)

f = open("vocabulary.txt", "w", encoding="utf8")
for word, count in vocabulary.most_common(1000):
print(word, file=f)   
```
After building the vocabulary, we used bag of words (BOW) representation for preparing the training, testing and validation dataset in which each row of the matrix illustrates each document, and each column shows the occurrences of the word in the vocabulary corresponding to that document.
```python
documents = []
labels = []
for f in os.listdir("aclImdb/smalltrain/pos"):
path = "aclImdb/smalltrain/pos/" + f
bow = read_document(path, vocabulary)
documents.append(bow)
labels.append(1)
for f in os.listdir("aclImdb/smalltrain/neg"):
path = "aclImdb/smalltrain/neg/" + f
bow = read_document(path, vocabulary)
documents.append(bow)
labels.append(0)

X = np.stack(documents)
Y = np.array(labels)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("train.txt.gz", data)
```    
## 2.2) Training & Analysing the model
The model was trained on the prepared document which contains bag of words (both for positive and negative) and their corresponding labels. It is found out that the training 
accuracy of the model is 80.56%.
For the analysis of the model, We sorted the weight matrix and used its indices to extract the top 10 positive and negative words from the vocabulary. We found out that some of the most impactful words on the predictions for negative reviews are waste, worst, awful and for positive reviews are loved, perfectly, brilliant etc. 
```Python
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
```
By using the scores matrix, we found the misclassified documents, which were 2430 in number, and sorted the scores matrix to find the top 10 documents which were misclassified by the model with higher confidence. Further analyzing the reviews in the documents, it can be said that due to sarcastic reviews or using too many negative words in the document (which is a positive review) as well as some positive review documents By using the scores matrix, we found the misclassified documents, which were 2430 in 
number, and sorted the scores matrix to find the top 10 documents which were misclassified by the model with higher confidence. Further analyzing the reviews in the 
documents, it can be said that due to sarcastic reviews or using too many negative words in the document (which is a positive review) as well as some positive review documents.
```python
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
print('Positive Missclassified reviews')
# top 10 positive missclassified documents
for tuple in np_mis_clsfd[-10:]:
	path = "aclImdb/smalltrain/neg/"
	f = open(path + documents[tuple[0]], encoding="utf8")
	text = f.read()
	f.close()
	print(text)
```
## 2.3) Evaluating the model with different variants
The test accuracy of the model is 80.39%, hence the model is not overfitting as the training and test accuracies are quite close. For underfitting, the accuracy is quite decent so if, even the underfitting happens it is going to be very small.

For collecting different results, we applied different techniques like stemming, removing stop words , increasing training and vocabulary size.For stemming we loaded ```stem``` function from ```Porter.py``` and directly applied it on the words before creating dictionary as well as while creating Bag of Words.
```Python 
from Porter.py import stem
#apply stemming on the word
word = stem(word)
````
For removing the stop words we have a file in our dataset which contains the most commonly used words in English Language. We removed them from our vocabulary because they don't add much meaning to the sentence.
```Python 
stopwords = open("aclImdb/stopwords.txt").read().splitlines()
for word in list(vocabulary):
    if word in stopwords:
        del vocabulary[word]
````

The best model that we found for the prediction is with 5000 vocabulary size, trained on large dataset without stemming ( as it decreases the test accuracy of the model ) and with removed stop words, with this model we got the training accuracy which is 85.60% with test accuracy equivalent to 83.42%. If we further increase the vocabulary size it makes the model to overfit the problem.
