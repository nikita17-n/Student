import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

e= pd.read_csv('student_dataset_detailed.csv')

# Filling Null Values

e['Age'] = e['Age'].fillna(e['Age'].mode()[0])
e['Study_Hours'] = e['Study_Hours'].fillna(e['Study_Hours'].mean())
e['Attendance'] = e['Attendance'].fillna(e['Attendance'].mode()[0])
e['Assignments_Submitted'] = e['Assignments_Submitted'].fillna(e['Assignments_Submitted'].median())
e['Extracurricular'] = e['Extracurricular'].fillna(e['Extracurricular'].mode()[0])
e['Previous_Score'] = e['Previous_Score'].fillna(e['Previous_Score'].mean())
e['Score'] = e['Score'].fillna(e['Score'].mean())
e['Result'] = e['Result'].fillna(e['Result'].mode()[0])
e['Gender'] = e['Gender'].fillna(e['Gender'].mode()[0])

# Dividing Features

X = e.drop(columns= 'Score')
y = e['Score']

# Splitting 

X_train, X_test, y_train,y_test = train_test_split(X, y,test_size= 0.2 , random_state= 0)

# Encoding

en = LabelEncoder()
X_train['Name'] = en.fit_transform(X_train['Name'])
X_train['Gender'] = en.fit_transform(X_train['Gender'])
X_train['Department']= en.fit_transform(X_train['Department'])
X_train['Extracurricular'] = en.fit_transform(X_train['Extracurricular'])
X_train['Result'] = en.fit_transform(X_train['Result'])

X_test['Name'] = en.fit_transform(X_test['Name'])
X_test['Gender'] = en.fit_transform(X_test['Gender'])
X_test['Department']= en.fit_transform(X_test['Department'])
X_test['Extracurricular'] = en.fit_transform(X_test['Extracurricular'])
X_test['Result'] = en.fit_transform(X_test['Result'])

# Normalization

s = MinMaxScaler(feature_range= (0,1))
X_train= s.fit_transform(X_train)
X_test= s.fit_transform(X_test)

# Now Training

c =LinearRegression()
c = c.fit(X_train,y_train)
pred = c.predict(X_test)

f = SVR()
g = f.fit(X_train,y_train)
h = g.predict(X_test)

print("Linear Regression:",pred)
print("Errors for Linear Regression:",mean_squared_error(pred, y_test))
print("RMSE:", np.sqrt(mean_squared_error(pred, y_test)))

print("\nSVR :",h)
print("Errors for SVR:",mean_squared_error(h, y_test))
print("RMSE:",np.sqrt(mean_squared_error(h, y_test)))
