import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd

# data = pd.read_excel (r'./dataset.xlsx')
# df = pd.DataFrame(data, columns= ['Ticket'])
# print (df)

# print (df)


def extract_data(filename):
    data = pd.read_excel(filename)
    features = [row[2] for row in data.values.astype('U')]
    # features = []
    # for row in data.values:
    #     feature = {
    #         'Title': row[1],
    #         'Body': row[2],
    #     }
    #     features.append(feature)

    labels = [row[3] for row in data.values]

    return features, labels


x_train, y_train = extract_data('./temp/dataset1.xlsx')
x_test, y_test = extract_data('./temp/dataset2.xlsx')
x_test1, y_test1 = extract_data('./temp/dataset_test.xlsx')


# vectorizer = DictVectorizer()
vectorizer = CountVectorizer()

vectorizer.fit(x_train)

vector = vectorizer.fit_transform(x_train).toarray()

# print(vectorizer.get_feature_names())
print('Vector Len: ')
print(len(vector))
# print('Vector Words: ')
# print(np.sum(vector))


# Build the classifier
clf = MultinomialNB(alpha=.01)


#  Train the classifier
clf.fit(vector, y_train)

# Get the test vectors
vectors_test = vectorizer.transform(x_test)

# Predict and score the vectors
pred = clf.predict(vectors_test)
acc_score = metrics.accuracy_score(y_test, pred)
f1_score = metrics.f1_score(y_test, pred, average='macro')

print('Total accuracy classification score: {}'.format(acc_score))
print('Total F1 classification score: {}'.format(f1_score))


vectors_test = vectorizer.transform(x_test1)

# Predict and score the vectors
pred = clf.predict(vectors_test)
acc_score = metrics.accuracy_score(y_test1, pred)
f1_score = metrics.f1_score(y_test1, pred, average='macro')

print('Total accuracy classification score: {}'.format(acc_score))
print('Total F1 classification score: {}'.format(f1_score))
