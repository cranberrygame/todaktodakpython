import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn import model_selection
#
from sklearn.linear_model import LogisticRegression

x_data = np.load('titanic_x_data.npy')
y_data = np.load('titanic_y_data.npy')
print(x_data.shape) #
print(y_data.shape) #
print(x_data[:5])
print(y_data[:5]) #

##from sklearn.preprocessing import StandardScaler
##scaler = StandardScaler()
##scaler.fit(x_data)
##x_data_scaled = scaler.transform(x_data)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=0.33)

#estimator = LogisticRegression()
estimator = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
estimator.fit(x_train, y_train)

print('estimator.coef_:', estimator.coef_) 
'''
estimator.coef_: [[ 0.29864911 -1.09809075]
 [-0.31026732  0.06349747]
 [-0.21938531  0.80216872]]
'''
print('estimator.intercept_:', estimator.intercept_) #

#'''
y_predict = estimator.predict(x_train)
score = metrics.accuracy_score(y_train, y_predict) #classification
print('train score: ', score)
y_predict = estimator.predict(x_test)
score = metrics.accuracy_score(y_test, y_predict)
print('test score: ', score)
#'''
'''
score = estimator.score(x_train, y_train) #metrics.r2_score (regression) or metrics.accuracy_score (classification) #내부에서 predict
print('train score: ', score)
score = estimator.score(x_test, y_test)
print('test score: ', score)
'''

print(x_test[:2])
'''
[[ 1.08412616]
 [-0.50335834]]
'''
y_predict = estimator.predict(x_test[:2])
print(y_predict) #
for y1, y2 in zip(y_test, y_predict):
    print(y1, y2, y1==y2)
