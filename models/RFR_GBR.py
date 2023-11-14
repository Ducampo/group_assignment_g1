import pandas as pd
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error




def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading training/test data")
    train = pd.DataFrame.from_records(json.load(open('train.json'))).fillna("")
    test = pd.DataFrame.from_records(json.load(open('test.json'))).fillna("")
    logging.info("Splitting validation")
    train, val = train_test_split(train, stratify=train['year'], random_state=123)
    featurizer = ColumnTransformer(
        transformers=[("title", TfidfVectorizer(max_features=1000), "title"),("abstract", TfidfVectorizer(max_features=2000), "abstract")],
        remainder='drop')
    RFR = make_pipeline(featurizer, RandomForestRegressor( max_features= "sqrt"))
    #GBR = make_pipeline(featurizer, GradientBoostingRegressor(max_features= "log2"))
    logging.info("Fitting models")
    RFR.fit(train.drop('year', axis=1), train['year'].values)
    #GBR.fit(train.drop('year', axis=1), train['year'].values)
    logging.info("Evaluating on validation data")
    err = mean_absolute_error(val['year'].values, RFR.predict(val.drop('year', axis=1)))
    logging.info(f"Mean baseline MAE: {err}")
    #err = mean_absolute_error(val['year'].values, GBR.predict(val.drop('year', axis=1)))
    #logging.info(f"GBR regress MAE: {err}")
    logging.info(f"Predicting on test")
    pred = RFR.predict(test)
    #pred = GBR.predict(test)
    test['year'] = pred.round()
    logging.info("Writing prediction file")
    test.to_json("predicted.json", orient='records', indent=2)
    
main()

