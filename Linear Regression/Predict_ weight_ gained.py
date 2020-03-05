import pandas as pd
import matplotlib.pyplot  as plt
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
