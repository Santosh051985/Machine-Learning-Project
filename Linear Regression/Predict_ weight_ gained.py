# Import Pandas library for data manipulation of raw
import pandas as pd
# Import matplotlib library for Data Visualization of data
import matplotlib.pyplot  as plt
#Import numpy for numerical calculation on dataset ##
import numpy as np
df = pd.read_csv("calories_consumed.csv")
df.head()
df.describe()
df.shape
df.info()
import seaborn as seabornInstance
df.plot(x='Calories Consumed', y='Weightgained_grams', style='o')  
plt.title('Calories Consumed vs Weightgained_grams')  
plt.xlabel('Calories Consumed')  
plt.ylabel('Weightgained_grams')  
plt.show()
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(df['Calories Consumed'])
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(df['Weightgained_grams'])
# Input dataset
X = df['Calories Consumed'].values.reshape(-1,1)
# Output or Predicted Value of data
y = df['Weightgained_grams'].values.reshape(-1,1)
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
#training the algorithm
#To retrieve the intercept:
print('Intercept Vale is ' , regressor.intercept_)
#For retrieving the slope:
print("Coefficient value is  ",regressor.coef_)

y_pred = regressor.predict(X_test)
df_p = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df_p
df_p.head(3)
df1 = df_p.head(3)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.splt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()how()
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# calculating mse =18569.747619585934
mse = np.mean((y_pred - y_test)**2) 
print(mse)
# Calculate rmse score = 0.5118476943631607
regressor.score(X_test,y_test)
## Transformation  for input variables for getting better R^2 value for the model .

# Ridge Regression
from sklearn.linear_model import Ridge

## training the model

ridgeReg = Ridge(alpha=0.05, normalize=True)

ridgeReg.fit(X_train,y_train)

#To retrieve the intercept:
print('Intercept Vale is ' , ridgeReg.intercept_)
#For retrieving the slope:
print("Coefficient value is  ",ridgeReg.coef_)

pred_rig = ridgeReg.predict(X_test)

##calculating mse =18569.747619585934
mse = np.mean((y_pred - y_test)**2)
print(mse)

## calculating score  =  0.5770007444041023
ridgeReg.score(X_test,y_test) 
### Lasso Regression
from sklearn.linear_model import Lasso

lassoReg = Lasso(alpha=0.05, normalize=True)

lassoReg.fit(X_train,y_train)

pred_lasso = lassoReg.predict(X_test)

# calculating mse = 18569.747619585934
mse = np.mean((y_pred - y_test)**2) 
print(mse)
#Calculate R^2 Error = 0.5125716544323572
lassoReg.score(X_test,y_test)
