# utf-8

import pandas as pd
import numpy as np

from mymodule.adaline import AdalineGD


"""
load data
"""
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values


"""
fit with ADALine
"""
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X,y)

