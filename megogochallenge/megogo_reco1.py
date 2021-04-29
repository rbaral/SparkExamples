'''
ref:https://www.kaggle.com/vchulski/tutorial-collaborative-filtering-with-pyspark
'''

import numpy as np # linear algebra
import pandas as pd # datasets processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc


full_df = pd.read_csv('datasets/megogochallenge/train_data_full.csv')
full_df.head()