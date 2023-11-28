import json
import pandas as pd
import numpy as np
import tensorflow as tf
import gensim.downloader as api
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Function to preprocess text: remove punctuation and lowercase
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = text.translate(str.maketrans('', '', string.punctuation))  
    return text

# Load data
train = pd.DataFrame.from_records(json.load(open('train.json')))
test = pd.DataFrame.from_records(json.load(open('test.json')))
train.fillna('', inplace=True)
test.fillna('', inplace=True)
train = train.drop(["editor"], axis=1)
train['ENTRYTYPE'] = pd.Categorical(train['ENTRYTYPE'])
test['ENTRYTYPE'] = pd.Categorical(test['ENTRYTYPE'])
train['year'] = pd.to_numeric(train['year'], errors="coerce")
test["author"] = test["author"].str.join(", ")
train["author"] = train["author"].str.join(", ")

# Preprocess the text data
train['abstract'] = train['abstract'].apply(preprocess_text)
train['title'] = train['title'].apply(preprocess_text)
train['author'] = train['author'].apply(preprocess_text)
train['publisher'] = train['publisher'].apply(preprocess_text)
test['abstract'] = test['abstract'].apply(preprocess_text)
test['title'] = test['title'].apply(preprocess_text)
test['author'] = test['author'].apply(preprocess_text)
test['publisher'] = test['publisher'].apply(preprocess_text)
# train['stratify_col'] = train['year'].astype(str) + "_" + train['ENTRYTYPE']

# Split data
train, val = train_test_split(train, stratify=train['ENTRYTYPE'], random_state=153)

# Load pre-trained Word2Vec model
w2v_model = api.load('word2vec-google-news-300')

# Function to create document vectors
def document_vector(word2vec_model, doc):
    doc = [word for word in doc.split() if word in word2vec_model.key_to_index]
    if len(doc) == 0:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word2vec_model[doc], axis=0)

# Apply function to 'abstract' text data
train['abstract_vector'] = train['abstract'].apply(lambda x: document_vector(w2v_model, x))
val['abstract_vector'] = val['abstract'].apply(lambda x: document_vector(w2v_model, x))

# Apply function to 'train' text data
train['title_vector'] = train['title'].apply(lambda x: document_vector(w2v_model, x))
val['title_vector'] = val['title'].apply(lambda x: document_vector(w2v_model, x))

# TF-IDF transformation for other features
featurizer = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(max_features=6000), "title"),
        ("author", TfidfVectorizer(max_features=6000), "author"),
        ("publisher", TfidfVectorizer(max_features=6000), "publisher"),
        ("abstract", TfidfVectorizer(max_features=6000), "abstract"),
        ("ENTRYTYPE", OneHotEncoder(), ["ENTRYTYPE"])
    ],
    remainder='drop'
)

X_train_tfidf = featurizer.fit_transform(train).toarray()
X_val_tfidf = featurizer.transform(val).toarray()

# Convert Word2Vec vectors into a format suitable for training
X_train_w2v = np.array(list(train['abstract_vector']))
X_val_w2v = np.array(list(val['abstract_vector']))

# Combine Word2Vec and TF-IDF features
X_train_combined = np.concatenate((X_train_tfidf, X_train_w2v), axis=1)
X_val_combined = np.concatenate((X_val_tfidf, X_val_w2v), axis=1)

# Normalize target variable (year)
y_train = train['year'].astype(float).values
y_val = val['year'].astype(float).values
y_mean = y_train.mean()
y_std = y_train.std()
y_train_normalized = (y_train - y_mean) / y_std
y_val_normalized = (y_val - y_mean) / y_std

# Build model function
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
                  loss='mean_absolute_error')
    return model

# Update the model with the new input shape
model = build_model(X_train_combined.shape[1])

# Train the model
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(X_train_combined, y_train_normalized, validation_data=(X_val_combined, y_val_normalized),
                    epochs=15, batch_size=32, callbacks=[callback])

# Predict and evaluate
y_val_pred_normalized = model.predict(X_val_combined).flatten()
y_val_pred = y_val_pred_normalized * y_std + y_mean
mae = mean_absolute_error(y_val, y_val_pred)
print(f"Mean Absolute Error on Validation Set: {mae}")
# Mean Absolute Error on Validation Set: 2.742465574971839