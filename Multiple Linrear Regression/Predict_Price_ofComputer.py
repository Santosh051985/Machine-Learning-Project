# Import pandas  library
import pandas as np
# Import numpy library
import numpy as np
import matplotlib.pyplot as plt
com_df = pd.read_csv("Computer_Data.csv")
com_df.head(10)
com_df1 =com_df.drop('Unnamed: 0',axis=1) 
com_df1.describe()
com_df1.shape
plt.hist(x= com_df1['price'])
cd_c= pd.get_dummies(com_df1['cd'], prefix='cd')
multi_c= pd.get_dummies(com_df1['multi'], prefix='multi')
premium_c =pd.get_dummies(com_df1['premium'], prefix='premium')
p_com_df =pd.concat([com_df1,cd_c,multi_c,premium_c], axis=1)
p_com_df1 = p_com_df.drop(['cd','cd1','multi','premium'], axis=1)
sns.set_style("whitegrid");
sns.pairplot(p_com_df1, hue="price", height=4);
plt.show()

