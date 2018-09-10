#Artifical Neural Network

#PART-1:Data Preprocessing
#________________________________________________________

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!
#___________________________________________________________________
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#initalising a ANN
classifier=Sequential()

#Adding a input layer and First hidden layer
classifier.add(Dense(units=6,activation='relu',init='uniform',input_dim=11))

#Adding 2nd hidden Layer
classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))

#Adding output layer
classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))
 
#compling ANN
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

#fitting ANN to training set
classifier.fit(X_train,y_train,batch_size=10,epochs=100) 


#Part-3 -Make prediction and evaluating results
#____________________________________________________________________

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)