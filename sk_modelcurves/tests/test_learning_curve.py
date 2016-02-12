# test cases to be added here for learning curves
# TODO: add useful test cases
import unittest
from sk_modelcurves.learning_curve import draw_learning_curve
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

print 'loading dataset...'
digits = datasets.load_digits()
X, y = digits.data, digits.target
knn = KNeighborsClassifier()
lr = LogisticRegression()
print 'drawing 1 curve...'
draw_learning_curve(knn, X, y, scoring='accuracy')
plt.show()
print 'drawing 2 curves...'
draw_learning_curve([knn, lr], X, y, scoring='accuracy', 
estimator_titles=['KNN', 'Logistic'])
plt.show()