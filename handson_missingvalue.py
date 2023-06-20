import numpy as np
import pandas as pd
# load the dataset
# use modified ethnic dataset
df = pd.read_csv(r'C:\3. Data preprocessing\DataSets-Data Pre Processing\DataSets\claimants.csv') # for doing modifications

# check for count of NA'sin each column
df.isna().sum()

# Create an imputer object that fills 'Nan' values
# Mean and Median imputer are used for numeric data (CLMSEX,CLMINSUR,SEATBELT,CLMAGE,CLMAGE)

# for Mean, Meadian, Mode imputation we can use Simple Imputer or df.fillna()
from sklearn.impute import SimpleImputer

# Mean Imputer 
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df["CLMINSUR"] = pd.DataFrame(mean_imputer.fit_transform(df[["CLMINSUR"]]))
df["SEATBELT"] = pd.DataFrame(mean_imputer.fit_transform(df[["SEATBELT"]]))
df["CLMAGE"] = pd.DataFrame(mean_imputer.fit_transform(df[["CLMAGE"]]))
df["CLMSEX"] = pd.DataFrame(mean_imputer.fit_transform(df[["CLMSEX"]]))
df["CLMSEX"].isna().sum()
df["CLMINSUR"].isna().sum()
df["SEATBELT"].isna().sum()
df["CLMAGE"].isna().sum()

df.isna().sum()
