#### zero variance and near zero variance ######
import pandas as pd
import numpy as np
data = pd.read_csv(r"C:\3. Data preprocessing\DataSets-Data Pre Processing\DataSets\Z_dataset.csv")
data.head()
# If the variance is low or close to zero, then a feature is approximately 
# constant and will not improve the performance of the model.
# In that case, it should be removed. 

data.var() # variance of numeric variables
data.var() == 0
