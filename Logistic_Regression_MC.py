from sklearn.model_selection import StratifiedShuffleSplit
import random
from sklearn import datasets
import numpy as np


def logistic_regression_multi(X_l, y_l, iter, w, l2, eta):
  m = len(X_l)
  for i in range(iter):
    step = X_l.dot(w)
    pred = softmax(step)
    gradients = 2/m * X_l.T.dot(pred - y_l) + (2 * l2 * w)
    w -= eta * gradients
  return w

def softmax(z):
    softed = []
    for item in z:
      e = np.exp(item - np.max(item))  # Numerically stable softmax
      soft = e / np.sum(e)
      softed.append(soft)
    return np.array(softed)

def one_hot_encode(y):
  classes = {
      0: [1,0,0],
      1: [0,1,0],
      2: [0,0,1]
  }
  y_ohe = []
  for i in y:
    y_ohe.append(classes[i[0]])
  return y_ohe

def inverse_ohe(y):
  y_c = []
  for i in y:
    max_value = max(i)
    max_index = list(i).index(max_value)
    y_c.append(max_index)
  return y_c

def shuffle_together(X, y):
  split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
  for train_index, test_index in split.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
  return X_train, y_train, X_test, y_test

def confusion_matrix(y_true, y_pred):
  num_classes = 3
  matrix = [[0] * num_classes for _ in range(num_classes)]
  for i in range(len(y_true)):
    true_class = y_true[i]
    pred_class = y_pred[i]
    matrix[true_class][pred_class] += 1
  return matrix

def accuracy(y_true, y_pred):
  good, wrong = 0, 0
  for i in range(len(y_true)):
    if y_true[i] == y_pred[i]:
      good += 1
    else:
      wrong += 1
  return good / len(y_true)

iris = datasets.load_iris()
list(iris.keys())

X = iris["data"]
X_b = np.c_[np.ones((len(X), 1)), X]
y = iris["target"]
y = y.reshape(-1, 1)

X_train, y_train, X_test, y_test = shuffle_together(X_b, y)

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

w = np.random.randn(5,3)
iter = 2000
eta = 0.01
l2 = 0.01

w = logistic_regression_multi(X_train, y_train, iter, w, l2, eta)

y_pred = softmax(X_test.dot(w))
y_r = inverse_ohe(y_test)
y_pred = inverse_ohe(y_pred)

print(confusion_matrix(y_r, y_pred))
print(accuracy(y_r, y_pred))
