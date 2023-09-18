# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: The first step is to prepare the data for the model. This involves cleaning the data, handling missing values and outliers, and transforming the data into a suitable format for the model.
2.Split the data: Split the data into training and testing sets. The training set is used to fit the model, while the testing set is used to evaluate the model's performance.
3.Define the model: The next step is to define the logistic regression model. This involves selecting the appropriate features, specifying the regularization parameter, and defining the loss function.
4.Train the model: Train the model using the training data. This involves minimizing the loss function by adjusting the model's parameters.
5.Evaluate the model: Evaluate the model's performance using the testing data. This involves calculating the model's accuracy, precision, recall, and F1 score.
6.Tune the model: If the model's performance is not satisfactory, you can tune the model by adjusting the regularization parameter, selecting different features, or using a different algorithm.
7.Predict new data: Once the model is trained and tuned, you can use it to predict new data. This involves applying the model to the new data and obtaining the predicted outcomes.
8.Interpret the results: Finally, you can interpret the model's results to gain insight into the relationship between the input variables and the output variable. This can help you understand the factors that influence the outcome and make informed decisions based on the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: HARISH RAGAVENDRA S
RegisterNumber:  212222230045
*/
```
```

import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
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
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
# placement data:
![Screenshot 2023-09-18 211424](https://github.com/harish-ragavendra-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114852180/404e0167-472d-4a12-9dbe-ee2a6f21eab4)
# salary data:
![Screenshot 2023-09-18 211424](https://github.com/harish-ragavendra-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114852180/92a7d226-14d9-47af-8f70-f88c703a416a)
# checking null function:
![Screenshot 2023-09-18 211858](https://github.com/harish-ragavendra-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114852180/7062dd32-a984-4eec-a0fa-1e922c87880a)
# Data Duplicate:
![Screenshot 2023-09-18 211934](https://github.com/harish-ragavendra-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114852180/de82a147-57f4-4b5c-bf17-4ffad8cbf453)
# print data:
![Screenshot 2023-09-18 212118](https://github.com/harish-ragavendra-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114852180/7e221a2f-1cfc-46a1-9192-be663501f3b7)
# Data Status:
![Screenshot 2023-09-18 212141](https://github.com/harish-ragavendra-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114852180/b8fb4496-875b-40c5-9048-43781984802b)
# y_prediction array:
![Screenshot 2023-09-18 212209](https://github.com/harish-ragavendra-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114852180/d84e9622-efe0-4a9d-bdff-0416f18b40f3)
# accuracy value:
![Screenshot 2023-09-18 212318](https://github.com/harish-ragavendra-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114852180/000b7c9d-38fe-4fac-977a-3c656e6778fa)
# confusion matrix:
![Screenshot 2023-09-18 212406](https://github.com/harish-ragavendra-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114852180/617dfbea-c988-45a9-99d5-d9f933141fa3)
# classification report:
![Screenshot 2023-09-18 212452](https://github.com/harish-ragavendra-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114852180/7f781ff9-fb4e-4e3f-8e7e-f733c4d31be1)
# prediction of LR:
![Screenshot 2023-09-18 212607](https://github.com/harish-ragavendra-25/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/114852180/352f4523-ba0f-44ee-8ffe-695322e37ee5)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
