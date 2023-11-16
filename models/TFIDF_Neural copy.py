import pandas as pd
import json
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error

train = pd.DataFrame.from_records(json.load(open('train.json')))
test = pd.DataFrame.from_records(json.load(open('test.json')))
train.fillna('', inplace=True)
test.fillna('', inplace=True)

train, val = train_test_split(train, stratify=train['year'], random_state=123)

featurizer = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(max_features=1000), "title"),
        ("abstract", TfidfVectorizer(max_features=1000), "abstract"),
    ],
    remainder='drop'
)

X_train = featurizer.fit_transform(train.drop(columns=['year'])).toarray()
X_val = featurizer.transform(val.drop(columns=['year'])).toarray()
y_train = train['year'].astype(float).values
y_val = val['year'].astype(float).values
X_test = featurizer.transform(test).toarray()

# Normalize target variable (year)
y_mean = y_train.mean()
y_std = y_train.std()
y_train_normalized = (y_train - y_mean) / y_std
y_val_normalized = (y_val - y_mean) / y_std

def build_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='mean_squared_error')
    return model

model = build_model(X_train.shape[1])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
model.fit(X_train, y_train_normalized, validation_data=(X_val, y_val_normalized), 
          epochs=10, batch_size=32, callbacks=[callback])

y_val_pred_normalized = model.predict(X_val).flatten()
y_val_pred = y_val_pred_normalized * y_std + y_mean

mae = mean_absolute_error(y_val, y_val_pred)
print(f"Mean Absolute Error on Validation Set: {mae}")

predictions_normalized = model.predict(X_test).flatten()
predictions = predictions_normalized * y_std + y_mean

predicted_json = [{'year': int(year)} for year in predictions]
with open('predicted_1.json', 'w') as f:
    json.dump(predicted_json, f)
