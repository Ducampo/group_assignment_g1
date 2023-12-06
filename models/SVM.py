import logging
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR  # Import SVR for Support Vector Regression
from sklearn.feature_extraction.text import LinearSVC
from sklearn.metrics import mean_absolute_error

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading training/test data")
    train = pd.DataFrame.from_records(json.load(open('r_train_file.json'))).fillna("")  
    test = pd.DataFrame.from_records(json.load(open('r_test_file.json'))).fillna("")
    logging.info("Splitting validation")
    train, val = train_test_split(train, stratify=train['year'], random_state=123)      #can we use less data or linear kernel, cuz the SVM might speed up alot of randomly delete 
    featurizer = ColumnTransformer(
        transformers=[("title_pre", TfidfVectorizer(), "title_pre"),
            ("abstract_low", TfidfVectorizer(), "abstract_low"),
            ("publisher_low", TfidfVectorizer(), "publisher_low")],
        remainder='drop')  # abstract feature was used as well as title

    dummy = make_pipeline(featurizer, DummyRegressor(strategy='mean'))
    svc_model = make_pipeline(featurizer, LinearSVC(random_state = 0, tol=1e-5))  # Using SVR for regression, increased C allows the model to fit more (risk overfitting)
    
    logging.info("Fitting models")
    dummy.fit(train.drop('year', axis=1), train['year'].values)
    svc_model.fit(train.drop('year', axis=1), train['year'].values)
    
    logging.info("Evaluating on validation data")
    err = mean_absolute_error(val['year'].values, dummy.predict(val.drop('year', axis=1)))
    logging.info(f"Mean baseline MAE: {err}")
    
    err = mean_absolute_error(val['year'].values, svm_model.predict(val.drop('year', axis=1)))
    logging.info(f"SVM regression MAE: {err}")
    
    logging.info(f"Predicting on test")
    pred = svm_model.predict(test)
    test['year'] = pred
    
    logging.info("Writing prediction file")
    test.to_json("predicted.json", orient='records', indent=2)

main()
