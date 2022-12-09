# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:31:21 2022

@author: hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits=load_digits()
digits.data
digits['feature_names']
digits['target']
digits['DESCR']
pd.DataFrame(digits['data'],columns=digits['feature_names'])
x=digits.data
y=digits.target
digits['data'][400]
digits.target[400]
plt.matshow(digits.images[400])
digits.data[11]
digits.images[11]
digits.target[11]
plt.matshow(digits.images[11])
plt.colormaps()
plt.set_cmap('Purples_r')
