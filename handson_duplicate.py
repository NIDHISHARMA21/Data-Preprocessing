################ Type casting #################
import pandas as pd

data = pd.read_csv(r"C:\3. Data preprocessing\DataSets-Data Pre Processing\DataSets\OnlineRetail.csv", encoding= 'unicode_escape')
data.dtypes

# Now we will convert 'float64' into 'int64' type. 
data.UnitPrice = data.UnitPrice.astype('int64')
data.dtypes

data.Quantity = data.Quantity.astype('float32')
data.dtypes

###############################################
### Identify duplicates records in the data ###
data = pd.read_csv(r"C:\3. Data preprocessing\DataSets-Data Pre Processing\DataSets\OnlineRetail.csv", encoding= 'unicode_escape')

duplicate = data.duplicated()
duplicate
sum(duplicate)

# Removing Duplicates
data1 = data.drop_duplicates()

###############################################

# Exploratory Data Analysis
# Measures of Central Tendency / First moment business decision
data1.Quantity.mean() # '.' is used to refer to the variables within object
data1.Quantity.median()
data1.Quantity.mode()

data1.UnitPrice.mean() # '.' is used to refer to the variables within object
data1.UnitPrice.median()
data1.UnitPrice.mode()

# Measures of Dispersion / Second moment business decision
data1.Quantity.var() # variance
data1.UnitPrice.std() # standard deviation

# Third moment business decision
data1.Quantity.skew()
data1.UnitPrice.skew()

# Fourth moment business decision
data1.Quantity.kurt()
data1.UnitPrice.kurt()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import numpy as np


#Quantity
plt.hist(data1.Quantity);plt.show() #histogram

plt.boxplot(data1.Quantity);plt.show() #boxplot

#UnitPrice
plt.hist(data1.UnitPrice);plt.show() #histogram

plt.boxplot(data1.UnitPrice);plt.show() #boxplot


