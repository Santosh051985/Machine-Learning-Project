import pandas as pd      ## For dataframe and functions on it
import numpy as np       ## For statistics and calculations
import matplotlib.pyplot as plt  ## For plotting

## Load the dataset
FChk = pd.read_csv("E:\\zz\\XLR_Data Science\\Assisgnments\\12. Decision Trees\\Fraud_check.csv")
FChk.head()    ## Gives the first five rows as the output

## Creating dummy variables for the given categorical data in the input
## Adding the dummy to the dataset
dummy1 = pd.get_dummies(FChk['Marital.Status'])    
FChk = pd.concat([FChk,dummy1],axis=1)             

dummy2 = pd.get_dummies(FChk['Urban'])
FChk = pd.concat([FChk,dummy2],axis=1)

dummy3 = pd.get_dummies(FChk['Undergrad'])
FChk = pd.concat([FChk,dummy3],axis=1)

## Dropping the categorical columns after creating their dummy variables
FraudCheck = FChk.drop(['Marital.Status','Urban','Undergrad'],axis=1)

## Creating a function to divide the data into risky and good using if else
def f(row):
    if row['Taxable.Income'] <=30000:
        val=("Risky")
        return val
    else:
        val=("Good")
        return val

## Creating a new column and storing the Risky and Good values in it
FraudCheck['FC'] = FraudCheck.apply(f, axis=1)

FraudCheck.FC.value_counts      ## 600 values of Risky and Good
colnames = list (FraudCheck.columns)    ## Gives the list of all the column names in the given dataframe

## Separating the given dataset for Predictors and Targets for Training and Test data
predictors = colnames[:10]    ## Considering the first 10 input columns in predictors
target = colnames[10]         ## The output column

## Training and Testing
from sklearn.model_selection import train_test_split   
train,test = train_test_split(FraudCheck, test_size = 0.2) ## Split train:test = 80:20

## Decision Tree Classifier for performing multi-class classification on a dataset.
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy')    ## quality of a split “entropy” for the information gain
model.fit(train[predictors],train[target])    ## Fit the model into predictors and target

preds = model.predict(test[predictors])   ## prediction model
pd.Series(preds).value_counts()     ## Gives the value of Risky and Good considered for prediction(unknown)
pd.crosstab(preds,test[target])     ## Getting the two-way table to understand the cross table

## Accuracy
np.mean(preds==test.FC)      ## 100%
