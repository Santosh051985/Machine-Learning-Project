import xgboost as xgb
import pandas as pd
ind_dia = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data",
                      names=["x1","x2","x3","x4","x5","x6","x7","x8","x9"])
ind_dia.head(10)
ind_dia.columns = ["Num_preg","Plasme_glucose","BP","TskinT","serum","BMI","DPF","AGE","CLASS"]

ind_dia.head(10)
# Checking any missing values were there in our data 
ind_dia.isnull().sum() # There were no missing values in our data


#### Implementing XGB classifier for predicting whether patient will have diabetes or not

from sklearn.model_selection import train_test_split
X,y=ind_dia.iloc[:,:8],ind_dia.iloc[:,8]
train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.28,random_state= 10)

### Preparing XGB classifier 
xgb1 = xgb.XGBClassifier(n_estimators=2000,learning_rate=0.3)
xgb1.fit(train_x,train_y)
train_pred = xgb1.predict(train_x)
import numpy as np
train_acc = np.mean(train_pred==train_y) # 90.94
test_pred = xgb1.predict(test_x)
test_acc = np.mean(test_pred==test_y) #0.75

# Variable importance plot 
from xgboost import plot_importance
plot_importance(xgb1)




