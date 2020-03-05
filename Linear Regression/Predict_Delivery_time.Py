import pandas as pd
import matplotlib.pyplot  as plt
import numpy as np
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
Deli_fr =pd.read_csv("delivery_time.csv")
Deli_fr.head()
Deli_fr.describe()
Deli_fr.info()
Deli_fr.plot(x='Sorting _Time', y='Delivery_Time', style='o')  
plt.title('Sorting _Time vs Delivery_Time')  
plt.xlabel('Sorting _Time')  
plt.ylabel('Delivery_Time')  
plt.show()
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(Deli_fr['Delivery_Time'])
# Input dataset
X = Deli_fr['Sorting _Time'].values.reshape(-1,1)
# Output or Predicted Value of data
y = Deli_fr['Delivery_Time'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor_d_t = LinearRegression()  
#training the algorithm
regressor_d_t.fit(X_train, y_train) 
#To retrieve the intercept:
print('Intercept Vale is ' , regressor_d_t.intercept_)

y_pred = regressor_d_t.predict(X_test)
df_p = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df_p
df_p.head(5)
df1 = df_p.head(3)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.58', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.58', color='black')
plt.show()

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


#For retrieving the slope:
print("Coefficient value is  ",regressor_d_t.coef_)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
