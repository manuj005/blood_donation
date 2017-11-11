import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

data1 = pd.read_csv('Blood_Train.csv')
data2 = pd.read_csv('Blood_Test.csv')
data1['ratio'] = data1['Months since First Donation'] / data1['Number of Donations']
data1['creative'] = data1['ratio'] - data1['Months since Last Donation']

data2['ratio'] = data2['Months since First Donation'] / data2['Number of Donations']
data2['creative'] = data2['ratio'] - data2['Months since Last Donation']

'''
# Checking accuracy of different models by dividing the training dataset further into training and testing

X = data1[['Months since Last Donation', 'Number of Donations', 'Total Volume Donated (c.c.)', 'Months since First Donation']]
Y = data1[['Made Donation in March 2007']]


X_Train, X_Test, Y_Train, Y_Test = cross_validation.train_test_split(X, Y, test_size = 0.2)

#clf = AdaBoostClassifier()
#clf = DecisionTreeClassifier(min_samples_split= 40)
#clf = svm.SVC()
#clf = LogisticRegression()
clf = GaussianNB()
clf.fit(X_Train, Y_Train)
pred = clf.predict(X_Test)
accu = accuracy_score(pred, Y_Test)
'''
filter1 = data1[data1['Months since Last Donation'] < 30]
filter2 = filter1[filter1['Number of Donations'] < 30]

A_feature = filter2[['creative', 'Months since Last Donation', 'Number of Donations', 'Months since First Donation']]
A_label = filter2[['Made Donation in March 2007']]

#plt.scatter(A_feature['ratio'], data1['Months since Last Donation'])
#plt.show()

test = data2[['creative', 'Months since Last Donation', 'Number of Donations', 'Months since First Donation']]
#test['ratio'] = test['Months since First Donation'] / test['Number of Donations']

#clf = GaussianNB()
clf = LogisticRegression()
#clf = DecisionTreeClassifier(min_samples_split= 40)
#clf = AdaBoostClassifier()
#clf = RandomForestClassifier()
#clf = svm.SVC(probability = True)

clf.fit(A_feature, A_label)
pred = clf.predict(test)

prob = clf.predict_proba(test)
val = pd.DataFrame(prob)
val.columns = ['0', '1']


output = data2[['Unnamed: 0']].copy()
output['Made Donation in March 2007'] = val[['1']]
output.to_csv('submission.csv')
print(output.head())
