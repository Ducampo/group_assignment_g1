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
r_test_eda_j = "add path to folder where you have stored the raw file"

# load preprocess EDA dataframe
pre_train_df = pd.read_json(r_train_eda_j)
pre_test_df = pd.read_json(r_test_eda_j)

# feature extraction: combine lowered columns both train/test dataset
pre_train_df["title_publisher_low"] = (
    pre_train_df["title_low"] + " " + pre_train_df["publisher_low"]
)
pre_train_df["title_publisher_low_len"] = pre_train_df["title_publisher_low"].apply(
    lambda x: len(x)
)

pre_test_df["title_publisher_low"] = (
    pre_test_df["title_low"] + " " + pre_test_df["publisher_low"]
)
pre_test_df["title_publisher_low_len"] = pre_test_df["title_publisher_low"].apply(
    lambda x: len(x)
)

# feature extraction: combine preprocess columns
pre_train_df["title_publisher_pre"] = (
    pre_train_df["title_n_stop_en"] + " " + pre_train_df["publisher_low"]
)
pre_train_df["title_publisher_len"] = pre_train_df["title_publisher_pre"].apply(
    lambda x: len(x)
)

pre_test_df["title_publisher_pre"] = (
    pre_test_df["title_n_stop_en"] + " " + pre_test_df["publisher_low"]
)
pre_test_df["title_publisher_len"] = pre_test_df["title_publisher_pre"].apply(
    lambda x: len(x)
)

"""
Identify different type of languages and each distribution based on different columns 
(Most accurate title+publisher)
"""


# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        # Return 'unknown' if language detection fails
        return "unknown"


"""
!NOTE Only run if necessary it takes >30' :S Find output under file poc_train_eda_j/poc_test_eda_j
"""
##
# title_publisher_low lang detect
# pre_train_df['lang_tit_publ_low'] = pre_train_df['title_publisher_low'].apply(detect_language)
print(
    f"Languages based on combination of tit+publi_low: \n{pre_train_df['lang_tit_publ_low'].value_counts()}"
)
# pre_test_df['lang_tit_publ_low'] = pre_test_df['title_publisher_low'].apply(detect_language)
print(
    f"Languages based on combination of tit+publi_low: \n{pre_test_df['lang_tit_publ_low'].value_counts()}"
)
"""
Languages based on combination of tit+publi_low train: 
en         62644
fr          2260
it           505
ca           121
da            72
pt            62
ro            50
sv            36
af            29
no            26
nl            22
tl            21
de            16
es            14
id            12
sq             5
et             5
fi             2
hr             2
cy             2
unknown        2
lv             2
lt             2
hu             1
vi             1
Name: lang_tit_publ_low, dtype: int64
"""
# title_low lang detect
# pre_train_df["languguage_detect"] = pre_train_df['title_low'].apply(detect_language)
print(
    f"Languages based on title_low: \n{pre_train_df['languguage_detect'].value_counts()}"
)
# pre_test_df["languguage_detect"] = pre_test_df['title_low'].apply(detect_language)
print(
    f"Languages based on title_low: \n{pre_test_df['languguage_detect'].value_counts()}"
)
"""
Languages based on title_low train: 
en         60387
fr          2454
it          1169
ca           448
da           344
ro           291
af           132
nl           102
no           101
es            88
sv            74
pt            70
tl            68
id            62
et            38
de            21
fi            11
sq             9
hr             6
pl             6
sl             6
lt             5
cy             5
sw             5
unknown        4
so             4
tr             2
sk             1
lv             1
Name: languguage_detect, dtype: int64
"""
# abstract_low lang detect
# pre_train_df["lang"] = pre_train_df['abstract_low'].apply(detect_language)
print(f"Languages based on abstract_low: \n{pre_train_df['lang'].value_counts()}")
# pre_test_df["lang"] = pre_test_df['abstract_low'].apply(detect_language)
print(f"Languages based on abstract_low: \n{pre_test_df['lang'].value_counts()}")

""" 
Languages based on abstract_low train: 
unknown    33644
en         30923
fr          1232
zh-cn        106
ko             4
no             3
da             1
it             1
Name: lang, dtype: int64
"""

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


extract_df_metadata(pre_train_df)

# export dataframe to json file
# feature_df.to_json('poc_feature_df_j.json', orient='records', lines=False)
