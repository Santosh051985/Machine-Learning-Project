import pandas as pd
import matplotlib.pyplot as plt

company = pd.read_csv("file:///F:/ASSIGNMENTS/decesion tree/Company_Data.csv")

# GENERATING DUMMIES

company1 = pd.get_dummies(company['Urban'])
company1

company2 = pd.get_dummies(company['US'])
company2
# for combining the two data sets
company=pd.concat([company,company1,company2],axis=1)
company.drop(["Urban","US"],inplace=True,axis=1)

company.head()
company["Sales"].unique()
company.Sales.value_counts()

colnames = list(company.columns)
predictors = colnames[1:13]
target = colnames[0]

# Splitting data into training and testing data set

import numpy as np
from sklearn.model_selection import train_test_split

train,test = train_test_split(company,test_size = 0.2)

from sklearn.tree import  DecisionTreeClassifier
help(DecisionTreeClassifier)

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])

preds = model.predict(test[predictors])
type(preds)
pd.Series(preds).value_counts()

pd.crosstab(test[target],preds)


np.mean(preds==test.Sales)
