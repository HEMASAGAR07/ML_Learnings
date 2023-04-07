import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabeties_dataset =pd.read_csv("diabetes.csv")
print (diabeties_dataset.head())
print(diabeties_dataset.describe())
print(diabeties_dataset['Outcome'].value_counts())
print(diabeties_dataset.groupby('Outcome').mean())
X=diabeties_dataset.drop('Outcome', axis=1)
Y=diabeties_dataset['Outcome']
scalar=StandardScaler()
standardised_data=scalar.fit_transform(X)
X=standardised_data
print(X)
print(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)

#training thr data
classifier=svm.SVC(kernel='linear')
#fitting training data into classifier
classifier.fit(X_train,Y_train)
#Evoluting our model
X_train_prediction=classifier.predict(X_train)
accuracy_prediction_train=accuracy_score(X_train_prediction,Y_train)
print(accuracy_prediction_train)
#test data
X_test_prediction=classifier.predict(X_test)
accuracy_prediction_test=accuracy_score(X_test_prediction,Y_test)
print(accuracy_prediction_test)

#evalution system

input_of_the_attributes=(5,116,74,0,0,25.6,0.201,30)

input_array_as_numpy_array=np.asarray(input_of_the_attributes)
#reshaping as we are not taking last outcome
input_array_as_numpy_array_reshaped= input_array_as_numpy_array.reshape(1,-1)
standadized_output=scalar.transform(input_array_as_numpy_array_reshaped)

prediction=classifier.predict(standadized_output)
print(prediction)

if (prediction==0):
    print("Person is not having diabeties")
else:
    print("The person is having diabeties")





