# Standardization and Normalization
import pandas as pd
import numpy as np

### Standardization
from sklearn.preprocessing import StandardScaler
d = pd.read_csv(r"C:\3. Data preprocessing\DataSets-Data Pre Processing\DataSets\Seeds_data.csv")

a = d.describe()
# Initialise the Scaler
scaler = StandardScaler()
# To scale data
df = scaler.fit_transform(d)
# Convert the array back to a dataframe
dataset = pd.DataFrame(df)
res = dataset.describe()


### Normalization
## load data set
d = pd.read_csv(r"C:\3. Data preprocessing\DataSets-Data Pre Processing\DataSets\Seeds_data.csv")
d.columns

a1 = d.describe()

# get dummies
d = pd.get_dummies(d, drop_first = True)

### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

df_norm = norm_func(d)
b = df_norm.describe()
