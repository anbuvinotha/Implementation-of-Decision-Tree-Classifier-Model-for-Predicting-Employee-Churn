# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:

/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Anbu Vinotha.s
RegisterNumber: 212223230015
*/

import pandas as pd
data = pd.read_csv("Employee (1).csv")
data.head()

data.info()
data.isnull().sum()
data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['salary'] = le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])


## Output:

## DATA HEAD 
![image](https://github.com/user-attachments/assets/cc268869-b1f4-4324-bff0-259a613268cf)

## DATASET INFO:

![image](https://github.com/user-attachments/assets/c76f9826-fd02-4df9-85ef-683674a66753)

## NULL DATASET:

## VALUES COUNT IN LEFT COLUMN:
![image](https://github.com/user-attachments/assets/3889c200-f831-44cf-b656-216aa30dc75d)

## DATASET TRANSFORMED HEAD:

![image](https://github.com/user-attachments/assets/ba07c1fa-0851-48e8-a4a3-2480c49ea808)

## X.HEAD:

![image](https://github.com/user-attachments/assets/06802e31-2d4d-49fe-9304-9567c8a11257)

## ACCURACY:

![image](https://github.com/user-attachments/assets/a76dc620-6f21-4296-8076-04ad802b1cba)

## DATA PREDICTION:

![image](https://github.com/user-attachments/assets/6ae72806-643f-4a2d-9918-73c9909aae85)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
