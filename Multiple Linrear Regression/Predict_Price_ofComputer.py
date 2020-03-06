import pandas as np
import numpy as np
import matplotlib.pyplot as plt
com_df = pd.read_csv("Computer_Data.csv")
com_df.head(10)
com_df1 =com_df.drop('Unnamed: 0',axis=1) 
com_df1.describe()
com_df1.shape
plt.hist(x= com_df1['price'])
