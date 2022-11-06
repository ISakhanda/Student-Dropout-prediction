import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import xgboost 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
#open file "train.csv" - data train
df_train = pd.read_csv("train.csv")
df_train.isnull().sum()
df_train['Category']= df_train['Category'].replace({'Graduate': 2, 'Dropout': 1, 'Enrolled': 0})
X = df_train[["Gender", 'Scholarship holder', 'Tuition fees up to date', "Application mode", 
"Curricular units 1st sem (approved)","Curricular units 1st sem (grade)", 
"Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)", 
"Curricular units 2nd sem (enrolled)"]]
y = df_train[["Category"]]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
#train model
model=XGBClassifier(max_depth=2, learning_rate=0.1, n_estimators=600, objective='binary:logistic', booster='gbtree')
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#The validation set. Has the targets. Use it to check overfitting.
df_validation = pd.read_csv("validation.csv")
df_validation['Category']= df_validation['Category'].replace({'Graduate': 2, 'Dropout': 1, 'Enrolled': 0})
X_test = df_validation[['Gender', 'Scholarship holder', 'Tuition fees up to date', "Application mode", 
"Curricular units 1st sem (approved)","Curricular units 1st sem (grade)", 
"Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)", 
"Curricular units 2nd sem (enrolled)"]]
y_test = df_validation[["Category"]]
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# The data to predict. TARGETS NOT PROVIDED. Use this one for the submission.
df_test= pd.read_csv("test.csv")
X_new = df_test[['Gender', 'Scholarship holder', 'Tuition fees up to date', 
"Application mode", "Curricular units 1st sem (approved)","Curricular units 1st sem (grade)", 
"Curricular units 2nd sem (approved)", 
"Curricular units 2nd sem (grade)", "Curricular units 2nd sem (enrolled)"]]
y_new = model.predict(X_new)
predictions_new= [round(value) for value in y_new]
df_test["Category"] = predictions_new
f_test['Category']= df_test['Category'].replace({2:'Graduate', 1:'Dropout', 0:'Enrolled'})
df = df_test[['Id','Category']]
df.to_csv('submission.csv',index=False)
