# Import pandas library for data manipulation 
import pandas as pd
# Import numpy library for calculation 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as train_test_split
df = pd.read_csv('sms_raw_NB.csv')
df
df.describe()
df.info()
