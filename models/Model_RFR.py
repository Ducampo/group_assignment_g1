import pandas as pd
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
#from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading training/test data")
    train = pd.DataFrame.from_records(json.load(open('poc_train_eda_lan_j (1).json'))).fillna("")
    test = pd.DataFrame.from_records(json.load(open('poc_test_eda_lan_j (1).json'))).fillna("")
    logging.info("Splitting validation")
    train, val = train_test_split(train, stratify=train['year'], random_state=123)
    featurizer = ColumnTransformer(
        transformers=[
            ("title", TfidfVectorizer(max_features=1000), "title"),
            ("abstract", TfidfVectorizer(max_features=1000), "abstract"),
            ("publisher",TfidfVectorizer(max_features=1000), "publisher"),
        ],
        remainder='drop'
    )
    #dummy = make_pipeline(featurizer, DummyRegressor(strategy='mean'))
    random_forest = make_pipeline(featurizer, RandomForestRegressor())

    logging.info("Fitting models")
    #dummy.fit(train.drop('year', axis=1), train['year'].values)
    random_forest.fit(train.drop('year', axis=1), train['year'].values)

    logging.info("Evaluating on validation data")
    #err_dummy = mean_absolute_error(val['year'].values, dummy.predict(val.drop('year', axis=1)))
    err_random_forest = mean_absolute_error(val['year'].values, random_forest.predict(val.drop('year', axis=1)))

    #logging.info(f"Mean baseline MAE: {err_dummy}")
    logging.info(f"Random Forest regressor MAE: {err_random_forest}")

    logging.info("Predicting on test")
    pred_random_forest = random_forest.predict(test)
    test['year'] = pred_random_forest

    logging.info("Writing prediction file")
    test.to_json("predicted_random_forest.json", orient='records', indent=2)

main()

INFO:root:Loading training/test data
INFO:root:Splitting validation
INFO:root:Fitting models
INFO:root:Evaluating on validation data
INFO:root:Mean baseline MAE: 7.8054390754858805
INFO:root:Random Forest regressor MAE: 3.351088605376291
INFO:root:Predicting on test
INFO:root:Writing prediction file
