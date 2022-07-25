"""Fraud_Detection.py

###

**AIM**

With the increasing technologies transaction of money has become lot easier but with pros comes con. Multiple crimes occur every minute related to fraud.

To overcome this issue machine learning approach has been introduced to predict what precautions can be taken to prevent these frauds

and which activites can raise new fraud activities based on hidden insights of previous tactics.

The dataset used is taken from public repository kaggle. The link to dataset is:  (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains details of costumers with whom fraudulent activities have happend along with instances.

###

Importing Libraries and Loading Dataset
"""

# Libraries
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

#importing libraries for evaluating model performance
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Loading the dataset
df=pd.read_csv('creditcard.csv')

# Preview of the dataset
df.head()

"""###Data Analysis"""

# shape of the dataset
df.shape

# check datatype of each attribute
df.info()

# checking the null values in the dataset
df.isnull().sum()

# checking the null values using heatmap
plt.figure(figsize=(12,6))
sns.heatmap(df.isnull())

total_transactions=len(df)
total_transactions

normal=len(df[df.Class==0])
fraudulent=len(df[df.Class==1])
fraud_percentage=round(fraudulent/normal*100,2)
print("Total Transactions are:",total_transactions)
print("Number of normal Transactions are:",normal)
print("Number of fraudulent Transactions are:",fraudulent)
print("Percentage of fraud Transactions are:",fraud_percentage)

min(df.Amount), max(df.Amount)

df.drop(['Time'], axis=1, inplace=True)

df.head()

df.shape

df.drop_duplicates(inplace=True)

df.shape

class_count_0,class_count_1 = df['Class'].value_counts()
print(class_count_0)
print(class_count_1)

# Separate the class
class_0 = df[df['Class'] == 0]
class_1 = df[df['Class'] == 1]
# print the shape of the class
print('class 0:', class_0.shape)
print('class 1:', class_1.shape)
class_1_over = class_1.sample(class_count_0, replace=True)
test_over = pd.concat([class_1_over, class_0], axis=0)

print("Total class of 0 and 1:",test_over['Class'].value_counts())
test_over['Class'].value_counts().plot(kind='bar',title='count(target)', color='green')

test_over.head()

"""###Splitting Train and Test Data"""

x=test_over.drop(['Class'], axis=1).values
y=test_over['Class'].values
x.shape, y.shape

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=0)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

"""###Modeling

#####Logistic Regression
"""

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

y_pred

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

sns.heatmap(cm,square=True,annot=True,cmap='Accent',fmt='g',cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')

#accuracy evaluation for logistic regressor
print('Accuracy score of the Logistic Regression model is:', accuracy_score(y_test, y_pred))
# calculate Precision
precision = precision_score(y_test, y_pred)
print('Precision Score:', precision)
# calculate recall
recall = recall_score(y_test, y_pred, average='binary')
print('Recall Score:', recall)
# calculate F1 Score
print('F1 score of the Logistic Regression model is', f1_score(y_test, y_pred))

""">logistic regressor has achieved an accuray of 94% which shows it is good classifier for this purpose.

Moreover precision and recall suggest that it classifies positive and negative classes accurately.

#####
 
K-nearest Neighbors
"""

from sklearn.neighbors import KNeighborsClassifier
n=5
knn=KNeighborsClassifier(n_neighbors=n)
knn.fit(X_train,y_train)

knn_pred=knn.predict(X_test)
knn_pred

cm=confusion_matrix(y_test,knn_pred)
cm

sns.heatmap(cm, square=True, annot=True, cmap='Pastel2_r', fmt='g', cbar=False)
plt.title('Confusion matrix')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')

print('Accuracy score of the KNN model is:', accuracy_score(y_test, knn_pred))

# calculate Precision
precision = precision_score(y_test, knn_pred)
print('Precision Score:', precision)

# calculate recall
recall = recall_score(y_test, knn_pred, average='binary')
print('Recall Score:', recall)

# calculate F1 Score
print('F1 score of the KNN model is', f1_score(y_test, knn_pred))

""">k-neighbor has achieved an accuray of 99% which shows it is the best classifier which can be taken into consideration for this problem.

Moreover precision and recall suggest that it classifies positive and negative classes accurately.

####
 
Support Vector Classifier
"""

from sklearn.svm import SVC
svm=SVC()
svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)
svm_pred

svm_pred[svm_pred==1].shape

y_test[y_test==1].shape

cm=confusion_matrix(y_test,svm_pred)
cm

sns.heatmap(cm, square=True, annot=True, cmap='gist_earth_r', fmt='g', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')

print('Accuracy score of SVM model is:', accuracy_score(y_test,svm_pred))

# calculate precision
precision=precision_score(y_test, svm_pred)
print('Precision Score:',precision)

# calculate recall
recall=recall_score(y_test, svm_pred, average='binary')
print('Recall Score:',recall)

# calculate F1 score
print('F1 score of the SVM model is', f1_score(y_test, svm_pred))

""">svc has achieved an accuray of 94% which shows it is good classifier for this purpose.

Moreover precision and recall suggest that it classifies positive and negative classes accurately.

####
 
Result

**COMPARISON OF TEST ACCURACY OF ALL ALGORITHMS**
"""

#test accuracy of all models
lr_test=accuracy_score(y_test,y_pred)
knn_test=accuracy_score(y_test,knn_pred)
svm_test=accuracy_score(y_test,svm_pred)


#plotting the accuracy graph 
plt.barh(y=['Logistic Regression','KNNClassifier','SupportVector'],width=[lr_test,knn_test,svm_test])
for index, value in enumerate([lr_test,knn_test,svm_test]):
    plt.text(value, index,
             str('%.2f' % value))
 
plt.show()

print(lr_test)
print(knn_test)
print(svm_test)

"""**Conclusion**

>After experimenting with 3 different machine learning models, it is seen that KNN comes up with highest accuracy.

Logistic regression and Support Vector classifiers are almost on same level of prediction.

KNN can be taken into practice for classifying real and fake transcations.
"""

