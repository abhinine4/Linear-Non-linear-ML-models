import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


def ldaLearn(X, y):
    class_labels = np.unique(y)
    means = np.zeros([class_labels.shape[0], X.shape[1]])

    for i in range(class_labels.shape[0]):
        m = np.mean(X[np.where(y == class_labels[i])[0],], axis=0)
        means[i,] = m

    covmat = np.cov(X.T)
    means = means.T
    return means, covmat


def qdaLearn(X, y):
    covmats = []
    class_labels = np.unique(y)
    means = np.zeros([class_labels.shape[0], X.shape[1]])

    for i in range(class_labels.shape[0]):
        m = np.mean(X[np.where(y == class_labels[i])[0],], axis=0)
        means[i,] = m
        covmats.append(np.cov(np.transpose(X[np.where(y == class_labels[i])[0],])))

    means = means.T
    return means, covmats


def ldaTest(means, covmat, Xtest, ytest):
  means = means.T
  out = np.zeros((Xtest.shape[0], means.shape[0]))
  covmat_det = np.linalg.det(covmat)
  convmat_inv = np.linalg.inv(covmat)
  den = np.sqrt((2 * pi) ** means.shape[1] * covmat_det)

  for j in range(means.shape[0]):
      out[:, j] = (np.exp(-0.5 * np.array(
          [np.dot(np.dot((Xtest[i, :] - means[j, :]),convmat_inv),
              (Xtest[i, :] - means[j, :]).T)
          for i in range(Xtest.shape[0])])))/den

  ypred = np.argmax(out, axis=1) + 1
  out = (ypred == ytest.reshape(len(ytest)))
  accuracy = len(np.where(out)[0])
  acc = float(accuracy) / (len(ytest))

  return acc, ypred


def qdaTest(means, covmats, Xtest, ytest):
    means = means.T
    out = np.zeros((Xtest.shape[0], means.shape[0]))

    for j in range(means.shape[0]):
        covmat_det = np.linalg.det(covmats[j])
        convmat_inv = np.linalg.inv(covmats[j])
        den = np.sqrt((2 * pi) ** means.shape[1] * covmat_det)
        out[:, j] = (np.exp(-0.5 * np.array(
            [np.dot(np.dot((Xtest[i, :] - means[j, :]),convmat_inv),
                    (Xtest[i, :] - means[j, :]).T)
            for i in range(Xtest.shape[0])])))/den

    ypred = np.argmax(out, axis=1) + 1
    out = (ypred == ytest.reshape(len(ytest)))

    accuracy = len(np.where(out)[0])
    acc = float(accuracy) / len(ytest)

    return acc, ypred


def learnOLERegression(X, y):
    XT = X.T
    XTX = np.matmul(XT, X)
    w = np.matmul(np.matmul(np.linalg.inv(XTX), XT), y)
    ypred = X.dot(w)
    error = y - ypred
    mse = np.mean(np.power(error, 2))
    return w


def learnRidgeRegression(X, y, lambd):
    XT = X.T
    XTX = np.matmul(XT, X) + lambd * np.identity(X.shape[1])
    w = np.matmul(np.matmul(np.linalg.inv(XTX), XT), y)
    return w


def testOLERegression(w, Xtest, ytest):
    ypred = Xtest.dot(w)
    error = ytest - ypred
    mse = np.mean(np.power(error, 2))
    return mse


def regressionObjVal(w, X, y, lambd):
    Xw = np.dot(X, w)
    Xw = np.array(Xw).reshape(242, 1)
    y_Xw = y - Xw
    y_Xw_t = np.transpose(y_Xw)
    e1 = np.dot(y_Xw_t, y_Xw)
    wt_w = np.dot(np.transpose(w), w)
    error = 0.5 * (e1[0][0]) + (0.5 * lambd) * wt_w

    Xt = np.transpose(X)
    Xt_X = np.dot(Xt, X)
    Xt_X_w = np.dot(Xt_X, w)
    Xt_X_w = np.array(Xt_X_w).reshape(65, 1)
    yt = np.transpose(y)
    yt_X_t = np.transpose(np.dot(yt, X))
    lambdw = np.array(lambd * w).reshape(65, 1)
    error_grad = (Xt_X_w - yt_X_t + lambdw)
    error_grad = error_grad.flatten()

    return error, error_grad


def mapNonLinear(x, p):
    x1 = np.array([x]).T
    Xp = np.ones_like(x1)
    for i in range(1, p + 1):
        Xp = np.hstack((Xp, x1 ** i))
    return Xp


# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'), encoding='latin1')

# LDA
means, covmat = ldaLearn(X, y)
ldaacc, ldares = ldaTest(means, covmat, Xtest, ytest)
print('LDA Accuracy = ' + str(ldaacc))
# QDA
means, covmats = qdaLearn(X, y)
qdaacc, qdares = qdaTest(means, covmats, Xtest, ytest)
print('QDA Accuracy = ' + str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5, 20, 100)
x2 = np.linspace(-5, 20, 100)
xx1, xx2 = np.meshgrid(x1, x2)
xx = np.zeros((x1.shape[0] * x2.shape[0], 2))
xx[:, 0] = xx1.ravel()
xx[:, 1] = xx2.ravel()

fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)

zacc, zldares = ldaTest(means, covmat, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zldares.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc, zqdares = qdaTest(means, covmats, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zqdares.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest.ravel())
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'), encoding='latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

w = learnOLERegression(X, y)
mle = testOLERegression(w, Xtest, ytest)

w_i = learnOLERegression(X_i, y)
mle_i = testOLERegression(w_i, Xtest_i, ytest)

print('MSE without intercept ' + str(mle))
print('MSE with intercept ' + str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k, 1))
mses3 = np.zeros((k, 1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i, y, lambd)
    mses3_train[i] = testOLERegression(w_l, X_i, y)
    mses3[i] = testOLERegression(w_l, Xtest_i, ytest)
    i = i + 1
fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(lambdas, mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas, mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k, 1))
mses4 = np.zeros((k, 1))
opts = {'maxiter': 20}  # Preferred value.
w_init = np.ones((X_i.shape[1], 1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args, method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l, [len(w_l), 1])
    mses4_train[i] = testOLERegression(w_l, X_i, y)
    mses4[i] = testOLERegression(w_l, Xtest_i, ytest)
    i = i + 1
fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(lambdas, mses4_train)
plt.plot(lambdas, mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize', 'Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas, mses4)
plt.plot(lambdas, mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize', 'Direct minimization'])
plt.show()

# Problem 5
pmax = 7
lambda_opt = 0.06  # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax, 2))
mses5 = np.zeros((pmax, 2))
for p in range(pmax):
    Xd = mapNonLinear(X[:, 2], p)
    Xdtest = mapNonLinear(Xtest[:, 2], p)
    w_d1 = learnRidgeRegression(Xd, y, 0)
    mses5_train[p, 0] = testOLERegression(w_d1, Xd, y)
    mses5[p, 0] = testOLERegression(w_d1, Xdtest, ytest)
    w_d2 = learnRidgeRegression(Xd, y, lambda_opt)
    mses5_train[p, 1] = testOLERegression(w_d2, Xd, y)
    mses5[p, 1] = testOLERegression(w_d2, Xdtest, ytest)

fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax), mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization', 'Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax), mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization', 'Regularization'))
plt.show()
