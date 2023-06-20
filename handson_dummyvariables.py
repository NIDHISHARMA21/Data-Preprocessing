import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# we use animal_category dataset
df = pd.read_csv(r"C:\3. Data preprocessing\DataSets-Data Pre Processing\DataSets\animal_category.csv")
df.columns # column names
df.shape # will give u shape of the dataframe
# drop Index column
df.drop(['Index'], axis=1, inplace=True)
df.dtypes
df_new = pd.get_dummies(df)# Create dummy variables
df_new_1 = pd.get_dummies(df, drop_first = True)
# we have created dummies for all categorical columns

##### One Hot Encoding works
df.columns

from sklearn.preprocessing import OneHotEncoder
# Creating instance of One Hot Encoder
enc = OneHotEncoder() # initializing method

enc_df = pd.DataFrame(enc.fit_transform(df.iloc[:, 0:]).toarray())


#######################
# Label Encoder
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()
# Data Split into Input and Output variables
X = df.iloc[:, 0:3]
y = df['Types']
y = df.iloc[:, 3:] # Alternative approach
df.columns
X['Animals']= labelencoder.fit_transform(X['Animals'])
X['Gender'] = labelencoder.fit_transform(X['Gender'])
X['Homly'] = labelencoder.fit_transform(X['Homly'])
### label encode y ###
y = labelencoder.fit_transform(y)
y = pd.DataFrame(y)
### we have to convert y to data frame so that we can use concatenate function
# concatenate X and y
df_new = pd.concat([X, y], axis =1)
## rename column name
df_new.columns
df_new = df_new.rename(columns={0:'Types'})
