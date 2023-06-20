import pandas as pd
import numpy as np
data = pd.read_csv(r"C:\3. Data preprocessing\DataSets-Data Pre Processing\DataSets\iris.csv")
data.head()
data.describe()
data = data.drop(["Unnamed: 0"], axis = 1)
#SepalLength
data['SepalLength_new'] = pd.cut(data['SepalLength'], bins=[min(data.SepalLength) - 1, 
                                                  data.SepalLength.mean(), max(data.SepalLength)], labels=["Low","High"])                                
data.head()
data.SepalLength_new.value_counts()
# data.SepalLength.value_counts()
# data.SepalWidth.value_counts()
# data.PetalLength.value_counts()
# data.PetalWidth.value_counts()
#SepalWidth
data['SepalWidth_new'] = pd.cut(data['SepalWidth'], bins=[min(data.SepalWidth) - 1, 
                                                  data.SepalWidth.mean(), max(data.SepalWidth)], labels=["Low","High"])

data.head()
data.SepalWidth_new.value_counts()

#PetalLength
data['PetalLength_new'] = pd.cut(data['PetalLength'], bins=[min(data.PetalLength) - 1, 
                                                  data.PetalLength.mean(), max(data.PetalLength)], labels=["Low","High"])

data.head()
data.PetalLength_new.value_counts()

#PetalWidth
data['PetalWidth_new'] = pd.cut(data['PetalWidth'], bins=[min(data.PetalWidth) - 1, 
                                                  data.PetalWidth.mean(), max(data.PetalWidth)], labels=["Low","High"])

data.head()
data.PetalWidth_new.value_counts()
data.Species.value_counts()


