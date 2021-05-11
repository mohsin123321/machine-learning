# Machine Learning Algorithms 
## 1) Logistic Regression
Logistic Regression implemented in Python from sratch. It uses gradient decent approach for the convergence of the model and uses the titanic data to calculate the probability of survivals.

## 2) Naive Bayes Algorithm
We used Naive Bayes classifier to predict the sentiments of the reviews, based on IMDB dataset , to predict them deemed negative or positive.
## 2.1) Data Preprocessing & Feature Extraction
Our dataset (attached in zip file __aclImdb.zip__) includes 50K reviews which are equally divided among positive and negative reviews from which 25K are for training, 12.5K are for testing and 12.5K are for validation. For making the computations and testing faster, we are considering small training data (approx. 6.2K for both positive and negative reviews) and later we will move to the original training data. Firstly, we build the vocabulary of words which counts 1K most common words which are used in the reviews both positive and negative. 
    
    ####for f in os.listdir("aclImdb/train/pos"):
      path = "aclImdb/train/pos/" + f
      words = read_document(path)
      vocabulary.update(words)
    for f in os.listdir("aclImdb/train/neg"):
      path = "aclImdb/train/neg/" + f
      words = read_document(path)
      vocabulary.update(words)

After building the vocabulary, we used bag of words (BOW) representation for preparing the training, testing and validation dataset in which each row of the matrix illustrates each document, and each column shows the occurrences of the word in the vocabulary corresponding to that document.
