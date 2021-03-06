import os
import collections
#from aclImdb.porter import stem

PUNCT = "!#$%&()'*+-/.,:;@?[]{}|^_`~<>=\"\\"
TABLE = str.maketrans(PUNCT, " " * len(PUNCT))

def read_stopwords():
    # loading stopwords from the file
    words = open("aclImdb/stopwords.txt").read().splitlines()
    return words

ignore = read_stopwords();

def read_document(filename):
    f = open(filename, encoding="utf8")
    text = f.read()
    f.close()
    text = text.lower()
    text = text.translate(TABLE)
    words = text.split()

    ignore_small_words = []
    for word in words:
        if len(word) > 2:
            ignore_small_words.append(word)

    return ignore_small_words


#     stem_words = []
#     for word in words:
#         stem_words.append(stem(word))
#     return stem_words


vocabulary = collections.Counter()
for f in os.listdir("aclImdb/train/pos"):
    path = "aclImdb/train/pos/" + f
    words = read_document(path)
    vocabulary.update(words)
for f in os.listdir("aclImdb/train/neg"):
    path = "aclImdb/train/neg/" + f
    words = read_document(path)
    vocabulary.update(words)

# removing the stopwords from dictionary
for word in list(vocabulary):
    if word in ignore:
        del vocabulary[word]

f = open("vocabulary.txt", "w", encoding="utf8")
for word, count in vocabulary.most_common(5000):
    print(word, file=f)
f.close()
