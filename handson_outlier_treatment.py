# 1st of all import all the packages
import pandas as pd
import numpy as np
import seaborn as sns
________________________________________________________________________________
# Load the data
df = pd.read_csv(r"C:\3. Data preprocessing\DataSets-Data Pre Processing\DataSets\boston_data.csv")
df.dtypes # finding data types
df.isna().sum() # finding is there any null values in the dataset or not but there is no null values in the dataset.
_____________________________

import sweetviz as sv

my_report = sv.analyze(df)
my_report.show_html() # Default arguments will generate to "SWEETVIZ_REPORT.html"
___________________________________________________
# let's find outliers
sns.boxplot(df.crim)
sns.boxplot(df.zn)
sns.boxplot(df.indus) # no outliers present
sns.boxplot(df.chas) # we leave this because it is a categorical variable and categorical variables has no outliers it's like 0's and 1's (Basically categories)
sns.boxplot(df.nox) # no outliers present
sns.boxplot(df.rm)
sns.boxplot(df.age) # no outliers present
sns.boxplot(df.dis)
sns.boxplot(df.rad) # no outliers present
sns.boxplot(df.tax) # no outliers present
sns.boxplot(df.ptratio)
sns.boxplot(df.black)
sns.boxplot(df.lstat)
sns.boxplot(df.medv)

# So, we have only 7 variables which has outliers
# 1) crim 2) zn 3) rm 4) dis 5) black 6) lstat 7) medv
________________________________________________________________________________
# 1. crim
# Detection of outliers (find limits for crim based on IQR)
IQR = df['crim'].quantile(0.75) - df['crim'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['crim'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['crim'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['crim'])
df_t = winsor.fit_transform(df[['crim']])
# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_
# lets see boxplot
sns.boxplot(df_t.crim)# we see no outiers
________________________________________________________________________________
# 2. zn
# Detection of outliers (find limits for zn based on IQR)
IQR = df['zn'].quantile(0.75) - df['zn'].quantile(0.25)
lower_limit = df['zn'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['zn'].quantile(0.75) + (IQR * 1.5)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['zn'])
df_t = winsor.fit_transform(df[['zn']])
# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_
# lets see boxplot
sns.boxplot(df_t.zn)# we see no outiers
________________________________________________________________________________
# 3. rm
# Detection of outliers (find limits for rm based on IQR)
IQR = df['rm'].quantile(0.75) - df['rm'].quantile(0.25)
lower_limit = df['rm'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['rm'].quantile(0.75) + (IQR * 1.5)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['rm'])
df_t = winsor.fit_transform(df[['rm']])
# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_
# lets see boxplot
sns.boxplot(df_t.rm)# we see no outiers
________________________________________________________________________________
# 4. dis
# Detection of outliers (find limits for dis based on IQR)
IQR = df['dis'].quantile(0.75) - df['dis'].quantile(0.25)
lower_limit = df['dis'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['dis'].quantile(0.75) + (IQR * 1.5)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['dis'])
df_t = winsor.fit_transform(df[['dis']])
# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_
# lets see boxplot
sns.boxplot(df_t.dis)# we see no outiers
________________________________________________________________________________
# 5. ptratio
# Detection of outliers (find limits for ptratio based on IQR)
IQR = df['ptratio'].quantile(0.75) - df['ptratio'].quantile(0.25)
lower_limit = df['ptratio'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['ptratio'].quantile(0.75) + (IQR * 1.5)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['ptratio'])
df_t = winsor.fit_transform(df[['ptratio']])
# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_
# lets see boxplot
sns.boxplot(df_t.ptratio)# we see no outiers

________________________________________________________________________________
# 6. black
# Detection of outliers (find limits for black based on IQR)
IQR = df['black'].quantile(0.75) - df['black'].quantile(0.25)
lower_limit = df['black'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['black'].quantile(0.75) + (IQR * 1.5)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['black'])
df_t = winsor.fit_transform(df[['black']])
# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_
# lets see boxplot
sns.boxplot(df_t.black)# we see no outiers
________________________________________________________________________________
# 7. lstat 
# Detection of outliers (find limits for lstat based on IQR)
IQR = df['lstat'].quantile(0.75) - df['lstat'].quantile(0.25)
lower_limit = df['lstat'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['lstat'].quantile(0.75) + (IQR * 1.5)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['lstat'])
df_t = winsor.fit_transform(df[['lstat']])
# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_
# lets see boxplot
sns.boxplot(df_t.lstat)# we see no outiers
________________________________________________________________________________
# 8. medv
# Detection of outliers (find limits for medv based on IQR)
IQR = df['medv'].quantile(0.75) - df['medv'].quantile(0.25)
lower_limit = df['medv'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['medv'].quantile(0.75) + (IQR * 1.5)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['medv'])
df_t = winsor.fit_transform(df[['medv']])
# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_
# lets see boxplot
sns.boxplot(df_t.medv)# we see no outiers
________________________________________________________________________________
############### Now Our data is Outlier free ###############