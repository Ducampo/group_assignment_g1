# Code finetune from following post: https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/
# Interesting post feature eng.: https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fwww.kaggle.com%2Fcode%2Fsolegalli%2Ffeature-engineering-and-model-stacking%2Fnotebook

# baseline imports:
import pandas as pd
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)  # changed the vectorized transformation from CountVectorizer to TFIDF
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# compare ensemble to each standalone models for regression
from numpy import mean
from numpy import std

# from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from matplotlib import pyplot


# get the dataset
def get_dataset():
    train = pd.DataFrame.from_records(
        json.load(open("poc_feature_train_eda_j.json"))
    ).fillna("")
    test = pd.DataFrame.from_records(
        json.load(open("poc_feature_test_eda_j.json"))
    ).fillna("")
    train, val = train_test_split(train, stratify=train["year"], random_state=123)
    X = train.drop("year", axis=1)
    y = train["year"].values
    # Preprocessing:
    featurizer = ColumnTransformer(
        transformers=[
            ("title_low", TfidfVectorizer(), "title_low"),
            ("publisher_low", TfidfVectorizer(), "publisher_low"),
            ("abstract_low", TfidfVectorizer(), "abstract_low"),
        ],
        remainder="drop",
    )
    X = featurizer.fit_transform(X)
    return X, y


# get a stacking ensemble of models
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(("rid", Ridge()))
    level0.append(("dum", DummyRegressor()))
    # level0.append(('svm', SVR()))
    # define meta learner model
    level1 = LinearRegression()
    # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    return model


# get a list of models to evaluate
def get_models():
    models = dict()
    models["knn"] = Ridge()
    models["cart"] = DummyRegressor()
    # models['svm'] = make_pipeline(featurizer,SVR())
    models["stacking"] = get_stacking()
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(
        model,
        X,
        y,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        error_score="raise",
    )
    return scores


# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print(">%s %.3f (%.3f)" % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
