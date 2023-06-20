# Normal Quantile-Quantile Plot

import pandas as pd

# Read data into Python
calorie = pd.read_csv(r"C:\3. Data preprocessing\DataSets-Data Pre Processing\DataSets\calories_consumed.csv")

import scipy.stats as stats
import pylab

# Checking Whether data is normally distributed
stats.probplot(calorie.Weight, dist="norm", plot=pylab)

stats.probplot(calorie.Calories, dist="norm", plot=pylab)

import numpy as np

# Transformation to make workex variable normal
stats.probplot((calorie.Weight*calorie.Weight), dist="norm", plot=pylab)
stats.probplot(np.sqrt(calorie.Weight), dist="norm", plot=pylab)
stats.probplot(1/calorie.Weight, dist="norm", plot=pylab)
stats.probplot(np.log(calorie.Weight), dist="norm", plot=pylab)

stats.probplot((calorie.Calories*calorie.Calories), dist="norm", plot=pylab)
stats.probplot(np.sqrt(calorie.Calories), dist="norm", plot=pylab)
stats.probplot(1/calorie.Calories, dist="norm", plot=pylab)
stats.probplot(np.log(calorie.Calories), dist="norm", plot=pylab)


