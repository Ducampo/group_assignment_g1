# %matplotlib inline # in case of working with jupyter notebook.


# Import all Libraries

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Declare document variables, make sure to update relative paths to match local file location
r_train_file = "add path to folder where you have stored the raw file"
r_test_file = "add path to folder where you have stored the raw file"

# raw data frames
r_train_eda_df = pd.read_json(r_train_file)
r_test_eda_df = pd.read_json(r_test_file)

# imbalance distribution most of the data is from 2010 to 2020
print(r_train_eda_df["year"])
r_train_eda_df["year"].hist()
plt.show()
