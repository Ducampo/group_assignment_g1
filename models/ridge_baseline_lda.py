# Include baseline topics lda and language
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


def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading training/test data")
    train = pd.DataFrame.from_records(
        json.load(
            open(
                "/content/drive/MyDrive/880083-M-6 Machine Learning/Assignments/model/lda/train_base_lda_en_topics.json"
            )
        )
    ).fillna("")
    test = pd.DataFrame.from_records(
        json.load(
            open(
                "/content/drive/MyDrive/880083-M-6 Machine Learning/Assignments/data/poc_feature_test_eda_lan_j.json"
            )
        )
    ).fillna("")
    logging.info("Splitting validation")
    train, val = train_test_split(train, stratify=train["year"], random_state=123)
    featurizer = ColumnTransformer(
        transformers=[
            ("title_low", TfidfVectorizer(), "title_low"),
            ("publisher_low", TfidfVectorizer(), "publisher_low"),
            ("abstract_low", TfidfVectorizer(), "abstract_low"),
            (
                "test_topic",
                OneHotEncoder(sparse=False),
                ["test_topic", "lang_tit_publ_low"],
            ),
        ],
        remainder="drop",
    )  # abstract feature was used as well as title
    dummy = make_pipeline(featurizer, DummyRegressor(strategy="mean"))
    ridge = make_pipeline(featurizer, Ridge())
    logging.info("Fitting models")
    dummy.fit(train.drop("year", axis=1), train["year"].values)
    ridge.fit(train.drop("year", axis=1), train["year"].values)
    logging.info("Evaluating on validation data")
    err = mean_absolute_error(
        val["year"].values, dummy.predict(val.drop("year", axis=1))
    )
    logging.info(f"Mean baseline MAE: {err}")
    err = mean_absolute_error(
        val["year"].values, ridge.predict(val.drop("year", axis=1))
    )
    logging.info(f"Ridge regress MAE: {err}")
    logging.info(f"Predicting on test")
    # pred = ridge.predict(test)
    # test['year'] = pred
    logging.info("Writing prediction file")
    # test.to_json("predicted.json", orient='records', indent=2)


main()
# INFO:root:Loading training/test data
# INFO:root:Splitting validation
# INFO:root:Fitting models
# /usr/local/lib/python3.10/dist-packages/sklearn/preprocessing/_encoders.py:868: FutureWarning: `sparse` was renamed to `sparse_output` in version 1.2 and will be removed in 1.4. `sparse_output` is ignored unless you leave `sparse` to its default value.
#   warnings.warn(
# /usr/local/lib/python3.10/dist-packages/sklearn/preprocessing/_encoders.py:868: FutureWarning: `sparse` was renamed to `sparse_output` in version 1.2 and will be removed in 1.4. `sparse_output` is ignored unless you leave `sparse` to its default value.
#   warnings.warn(
# INFO:root:Evaluating on validation data
# INFO:root:Mean baseline MAE: 7.714694216755001
# INFO:root:Ridge regress MAE: 3.932716889478392
# INFO:root:Predicting on test
# INFO:root:Writing prediction file
