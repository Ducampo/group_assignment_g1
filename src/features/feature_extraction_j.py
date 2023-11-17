# installing libraries in case not done already
#!pip install langdetect

# importing required libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import string
import nltk
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt")

## Declare document variables, make sure to update relative paths to match local file location
r_train_eda_j = "add path to folder where you have stored the raw file"

# load preprocess EDA dataframe
feature_df = pd.read_json(r_train_eda_j)
feature_df.info()

# feature extraction: combine existing columns
feature_df["title_publisher_pre"] = (
    feature_df["title_n_stop_en"] + " " + feature_df["publisher_low"]
)

feature_df["title_publisher_len"] = feature_df["title_publisher_pre"].apply(
    lambda x: len(x)
)

"""
Function to create dataframe metadata including col. Name, Description, Non-null and object type.
Parameters:
- df: pandas DataFrame
If no description assigned to col, will add no description.
"""


def extract_df_metadata(df):
    metadata = []

    for col in df.columns:
        if col == "ENTRYTYPE":
            description = "raw entrytype ?"
        elif col == "title":
            description = "raw paper title"
        elif col == "editor":
            description = "raw editor"
        elif col == "year":
            description = "raw year, target feature"
        elif col == "publisher":
            description = "raw publisher name. Note existing duplicates"
        elif col == "author":
            description = "raw author(s)"
        elif col == "abstract":
            description = 'prepro abstract, replaced null with ""'
        elif col == "title_len":
            description = "raw title column length"
        elif col == "abstract_len":
            description = "preprocess abstract column length"
        elif col == "title_low":
            description = "col title to lowercase"
        elif col == "publisher_low":
            description = "col publisher without null to lowercase"
        elif col == "abstract_low":
            description = "col abstract to lowercase"
        elif col == "title_pre":
            description = 'Del. punct. from title_low "-""." for dim'
        elif col == "title_pre_len":
            description = "col title_pre length"
        elif col == "abstract_pre":
            description = 'Del. punct. from abstract_low "-""." for dim'
        elif col == "abstract_pre_len":
            description = "col abstract_pre length"
        elif col == "title_n_stop_en":
            description = "Del. title_pre stopwords from english"
        elif col == "title_n_stop_en_len":
            description = "col length reduce dimensionality"
        elif col == "abstract_n_stop_en":
            description = "Del. abstract_pre stopwords from english"
        elif col == "abstract_n_stop_en_len":
            description = "col length reduce dimensionality"
        elif col == "title_publisher_pre":
            description = "concat title_n_stop_en & publi_low"
        elif col == "title_publisher_len":
            description = "col title_publisher_pre length"
        else:
            description = "Description not provided"

        metadata.append([col, description, df[col].count(), str(df[col].dtype)])

    metadata_df = pd.DataFrame(
        metadata, columns=["Column", "Description", "Non-Null Count", "Dtype"]
    )
    return metadata_df


extract_df_metadata(feature_df)

# export dataframe to json file
# feature_df.to_json('poc_feature_df_j.json', orient='records', lines=False)
