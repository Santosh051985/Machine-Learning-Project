import pandas as pd 
import numpy as np 
import seaborn as sns

salary_train=pd.read_csv("file:///F:/ASSIGNMENTS/support vector machines/SalaryData_Train(1).csv")
salary_test=pd.read_csv("file:///F:/ASSIGNMENTS/support vector machines/SalaryData_Test(1).csv")

#listing the top 5 head variables of given dataset
salary_train.head()
salary_test.head()

# discribing the given data set for EDA presentation(mean,std,min,max,count)
salary_train.describe()
salary_test.describe()


#sns.pairplot(data=salary data set== training and testing)

# visuvalization for salary data set with input and output variable by boxplot by training data set
sns.boxplot(x="Salary",y="age",data=salary_train,palette = "hls")
sns.boxplot(x="age",y="Salary",data=salary_train,palette = "hls")

# visuvalization for salary data set with input and output variable by boxplot by testing data set
sns.boxplot(x="Salary",y="age",data=salary_test,palette = "hls")
sns.boxplot(x="age",y="Salary",data=salary_test,palette = "hls")

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

salary_train_x=salary_train.iloc[:,0:12]
salary_train_y=salary_train.iloc[:,12]

salary_test_x=salary_test.iloc[:,0:12]
salary_test_y=salary_test.iloc[:,12]

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# kernel = linear
help(SVC)
model_linear=SVC(kernel = "linear")
model_linear.fit(salary_train_x,salary_train_y)
predict_linear_test=model.linear(salary_test_y)
np.mean(predict_lineat_test==salary_test_y)  ## Accuracy = 85%

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(salary_train_x,salary_train_y)
pred_test_rbf = model_rbf.predict(salary_test_y)

np.mean(pred_test_rbf==salary_test_y) # Accuracy = 97.016
