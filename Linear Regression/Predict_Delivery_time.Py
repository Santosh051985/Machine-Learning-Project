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
