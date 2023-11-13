# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## Program:

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by:MADHUMITHA M

Register Number:212222220020

import pandas as pd

df=pd.read_csv('/content/Placement_Data (1).csv')

df.head()

df1=df.copy()

df1=df1.drop(['sl_no','salary'],axis=1)

df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df1["gender"]=le.fit_transform(df1["gender"])

df1["ssc_b"]=le.fit_transform(df1["ssc_b"])

df1["hsc_b"]=le.fit_transform(df1["hsc_b"])

df1["hsc_s"]=le.fit_transform(df1["hsc_s"])

df1["degree_t"]=le.fit_transform(df1["degree_t"])

df1["workex"]=le.fit_transform(df1["workex"])

df1["specialisation"]=le.fit_transform(df1["specialisation"])

df1["status"]=le.fit_transform(df1["status"])

df1

x=df1.iloc[:,:-1]

x

y=df1["status"]

y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(solver='liblinear') #library for linear classification

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

y_pred

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,y_pred)

accuracy

from sklearn.metrics import confusion_matrix

confusion=confusion_matrix(y_test,y_pred)

confusion

from sklearn.metrics import classification_report

classification_report=classification_report(y_test,y_pred)

print(classification_report)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])



## Output:

![image](https://github.com/Madhumithamahendran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394403/3af0408f-a386-4f41-8465-01d98c66826a)

![image](https://github.com/Madhumithamahendran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394403/47267739-de0f-41d5-ba39-58411371f477)

![image](https://github.com/Madhumithamahendran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394403/79dbf8cc-22ea-4e29-be97-6431fd6a0e59)

![image](https://github.com/Madhumithamahendran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394403/809e24f8-3090-4bb2-bbc0-e09a3d04eec8)

![image](https://github.com/Madhumithamahendran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394403/626a8427-c21b-490b-9909-bdf745157731)

![image](https://github.com/Madhumithamahendran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394403/74432c4c-0f24-470f-bce4-e4cd48bfe633)

![image](https://github.com/Madhumithamahendran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394403/b229ac4d-c94a-4e39-8e45-ad90c3489482)

![image](https://github.com/Madhumithamahendran/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394403/0edd3256-bb3a-4ad6-bec0-5a755645f00e)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
