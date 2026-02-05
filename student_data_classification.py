import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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
print(e.to_string())
# Dividing Features

X = e.drop(columns= 'Result')
y = e['Result']

# Splitting 

X_train, X_test, y_train,y_test = train_test_split(X, y,test_size= 0.2 , random_state= 0)

# Encoding

en = LabelEncoder()
X_train['Name'] = en.fit_transform(X_train['Name'])
X_train['Gender'] = en.fit_transform(X_train['Gender'])
X_train['Department']= en.fit_transform(X_train['Department'])
X_train['Extracurricular'] = en.fit_transform(X_train['Extracurricular'])

X_test['Name'] = en.fit_transform(X_test['Name'])
X_test['Gender'] = en.fit_transform(X_test['Gender'])
X_test['Department']= en.fit_transform(X_test['Department'])
X_test['Extracurricular'] = en.fit_transform(X_test['Extracurricular'])

# Normalization

# s = MinMaxScaler(feature_range= (0,1))
# X_train= s.fit_transform(X_train)
# X_test= s.fit_transform(X_test)

# Now Training

c =LogisticRegression(max_iter = 1000)
c = c.fit(X_train,y_train)
pred = c.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
d = knn.fit(X_train,y_train)
pre = d.predict(X_test)

f = SVC()
g = f.fit(X_train,y_train)
h = g.predict(X_test)

print("Logistic Regression:",pred)
print("Accuracy of Logistic Regression :",accuracy_score(y_test,pred))

print("\nKNN: ",pre)
print("Accuracy  of KNN : ",accuracy_score(y_test,pre))

print("\nSVC :",h)
print("Accuracy of SVC:",accuracy_score(y_test,h))

# Checking For Overfitting  (For Logistic Regression)

y_train_pred = c.predict(X_train)
print("\nChecking For Overfitting of Logistic Regression :",accuracy_score(y_train,y_train_pred))

# (For KNN)

y_train_pre = d.predict(X_train)
print("Checking For Overfitting of KNN : ",accuracy_score(y_train,y_train_pre))

# (For SVC)

y_train_h = g.predict(X_train)
print("Checking For Overfitting of SVC : ",accuracy_score(y_train,y_train_h))

new = pd.DataFrame({
    'Student_ID' : ['45'],
    'Name' : 'Student_45',               #45,Student_45,Female,,IT,2,65,5,No,91,91,Pass
    'Gender' : "Female",
    'Age' : ['25'],
    'Department' : 'IT',
    'Study_Hours' : ["2"],
    'Attendance' : ['65'],
    'Assignments_Submitted' : ['5'],
    'Extracurricular' : 'No',
    'Previous_Score' :['91'],
    'Score' : ['91']

})

new['Name'] = en.fit_transform(new['Name'])
new['Gender'] = en.fit_transform(new['Gender'])
new['Department']= en.fit_transform(new['Department'])
new['Extracurricular'] = en.fit_transform(new['Extracurricular'])

new_pred = d.predict(new)
x_pre = d.predict(X_train)
print(x_pre)

results = pd.DataFrame(X_train.copy())   # keep your features
results["Actual"] = y_train             # if you have true labels
results["Predicted"] = x_pre         # add predictions

results.to_csv("prediction.csv", index=False)

print("\nPredictions are saved in \"prediction.csv\" file.")

