import pydotplus
from sklearn.externals.six import StringIO
import numpy as np

import graphviz
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()

test_idx = [0, 50, 100]

#  Trainind data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))

# print(iris.feature_names)
# print(iris.target_names)
# print(iris.data[0])
# print(iris.target[0])


dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     impurity=False)

# graph = graphviz.Source(dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")


print(test_data[1], test_target[1])
print(iris.feature_names, iris.target_names)
