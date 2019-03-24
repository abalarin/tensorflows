from sklearn import tree

'''
Table: "features"
[weight, texture]
weight: grams
texture: 0 = 'Bumpy', 1 = 'Smooth'

Table: "labels"
0: Apple
1: Orange
'''

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

#  [Weight, Texture] - [100g, Smooth]
predicitons = clf.predict([[100, 1]])

for prediction in predicitons:
    if prediction == 0:
        print('Apple')
    elif prediction == 1:
        print('Orange')
