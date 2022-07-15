# importing the dataset
import pandas as pd
 
data = pd.read_csv('C:/Users/Hp/Downloads/TNBC_survival.xlsx - Sheet1.csv')   

#Remove the column EmployeeNumber
data = data.drop(['Treatment given on relapse','Outcome_time','Survival ','event','relapse_time'], axis = 1) # A number assignment 
data=data.fillna(0)

data_copy=data.copy()
target = 'relapse'
category_col =['HPE','Stage','Tumor_Size','Surgery','Chemo_given_initially']


for col in category_col:
    dummy = pd.get_dummies(data_copy[col], prefix=col)
    data_copy = pd.concat([data_copy,dummy], axis=1)
    del data_copy[col]

#Split the data into independent 'X' and dependent 'Y' variables
X = data_copy.drop('relapse',axis=1)
Y = data_copy['relapse']

# Split the dataset into 75% Training set and 25% Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


#Use Random Forest Classification algorithm
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(X, Y)


# Saving the model
import pickle
pickle.dump(forest, open('tnbc_clf.pkl', 'wb'))
