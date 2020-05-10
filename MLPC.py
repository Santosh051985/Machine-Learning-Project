# Import library Numpy for numerical calculation ###
import numpy as np
# Import pandas for data manipulation of data ****
import pandas as pd
import matplotlib.pyplot as plt
# Read CSV File 
wbcd = pd.read_csv("C:\\Users\\HP\\Desktop\\wbcd.csv")
# Shows features of dataset
wbcd.columns
# Dropping the column which has not involve in calculation
wbcd.drop(["id"],axis=1,inplace=True) 
# Handle No missing values ######
wbcd.isnull().sum() 

wbcd.loc[wbcd.diagnosis=="B","diagnosis"] = 1
wbcd.loc[wbcd.diagnosis=="M","diagnosis"] = 0

X = wbcd.drop(["diagnosis"],axis=1)
Y = wbcd["diagnosis"]
plt.hist(Y)
wbcd.diagnosis.value_counts()

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y)
# Feature Scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(30,30))

mlp.fit(X_train,y_train)
prediction_train=mlp.predict(X_train)
prediction_test = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction_test))
np.mean(y_test==prediction_test)
np.mean(y_train==prediction_train)

