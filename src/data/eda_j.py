# %matplotlib inline # in case of working with jupyter notebook.


# import required libraries

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nltk


# Declare document variables, make sure to update relative paths to match local file location
r_train_file = "add path to folder where you have stored the raw file"
r_test_file = "add path to folder where you have stored the raw file"

# raw data frames
r_train_eda_df = pd.read_json(r_train_file)
# r_test_eda_df = pd.read_json(r_test_file)


# Descriptive view training dataset
r_train_eda_df.head()
r_train_eda_df.shape
r_train_eda_df.info()

## To be checked: https://www.kaggle.com/code/harshsingh2209/complete-guide-to-eda-on-text-data
r_train_eda_df.isnull().sum()
r_train_eda_df["title"][65909]
print(r_train_eda_df.iloc[65909])



## Descriptive analysis training dataset
r_train_eda_df.head()
r_train_eda_df.shape
r_train_eda_df.info()
r_train_eda_df.describe()

# To be checked: https://www.kaggle.com/code/harshsingh2209/complete-guide-to-eda-on-text-data #https://www.section.io/engineering-education/using-imbalanced-learn-to-handle-imbalanced-text-data/#prerequisites
r_train_eda_df.isnull().sum()
r_train_eda_df["title"][65909]
print(r_train_eda_df.iloc[65909])

""" 
!important Note:
- Not all text is in english, we may be able to get the language from the papers based on their publisher? (language not constant within publisher)
- publisher names are not always imputed the same way (Need to be cleaned)
- !important Some publishers have the year of publication in the name
"""
print(r_train_eda_df["publisher"].unique())
r_train_eda_df[["publisher"]].value_counts()

# imbalance distribution most of the data is from 2010 to 2020
print(r_train_eda_df["year"])
print(r_train_eda_df["year"].value_counts())

r_train_eda_df["year"].hist()

## data cleaning
train_df = r_train_eda_df

print(f"Before null iputation: {train_df['abstract'].isnull().sum()}\n")
# Replacing Null values to blank for abstract to be able to check the length
train_df["abstract"].fillna("", inplace=True)
print(train_df["abstract"].isnull().sum())
# print(train_df[train_df['abstract_len']==0])

# Adding columns lengths for title and abstract
train_df["title_len"] = train_df["title"].apply(lambda x: len(x))
train_df["abstract_len"] = train_df["abstract"].apply(lambda x: len(x))
train_df.info()
# Value range for columns ['title','abstract']
print(train_df["title_len"].max())
print(train_df["title_len"].min())
print(train_df["abstract_len"].max())
print(train_df["abstract_len"].min())

# add lowercase columns of interest ['title', 'publisher', 'abstract']
train_df["title_low"] = train_df["title"].str.lower()
train_df["publisher_low"] = train_df["publisher"].str.lower()
train_df["abstract_low"] = train_df["abstract"].str.lower()
train_df.head(20)
