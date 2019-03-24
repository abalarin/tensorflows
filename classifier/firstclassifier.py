from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
import random

from scipy.spatial import distance


def eucdistnace(a, b):
    return distance.euclidean(a, b)


class scrapyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predicitons = []

        for row in X_test:
            label = self.closest(row)
            predicitons.append(label)

        return predicitons

    def closest(self, row):
        best_distance = eucdistnace(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = eucdistnace(row, self.X_train[i])
            if dist < best_distance:
                best_index = i

        return self.y_train[best_index]

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

# from sklearn.neighbors import KNeighborsClassifier
my_classifier = scrapyKNN()
# my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit(X_train, y_train)

predicitons = my_classifier.predict(X_test)
# print(predicitons)

print accuracy_score(y_test, predicitons)
