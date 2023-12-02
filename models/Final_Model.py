import json
import string
import pandas as pd
import numpy as np
import tensorflow as tf
import gensim.downloader as api
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = text.translate(str.maketrans('', '', string.punctuation))  
    return text

# Load and preprocess data
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

# Apply text preprocessing
train['abstract'] = train['abstract'].apply(preprocess_text)
train['title'] = train['title'].apply(preprocess_text)
train['author'] = train['author'].apply(preprocess_text)
train['publisher'] = train['publisher'].apply(preprocess_text)
test['abstract'] = test['abstract'].apply(preprocess_text)
test['title'] = test['title'].apply(preprocess_text)
test['author'] = test['author'].apply(preprocess_text)
test['publisher'] = test['publisher'].apply(preprocess_text)

# Create Combined features
train['author_publication'] = train['author'] + ' ' + train['publisher']
train['title_abstract'] = train['title'] + ' ' + train['abstract']
test['author_publication'] = test['author'] + ' ' + test['publisher']
test['title_abstract'] = test['title'] + ' ' + test['abstract']


# Split data
train, val = train_test_split(train, stratify=train['year'], random_state=153)

# Load pre-trained Word2Vec model
w2v_model = api.load('word2vec-google-news-300')

# Function to create document vectors
def document_vector(word2vec_model, doc):
    doc = [word for word in doc.split() if word in word2vec_model.key_to_index]
    if len(doc) == 0:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word2vec_model[doc], axis=0)

# Apply function to create vectors
train['abstract_vector'] = train['abstract'].apply(lambda x: document_vector(w2v_model, x))
val['abstract_vector'] = val['abstract'].apply(lambda x: document_vector(w2v_model, x))

# TF-IDF transformation
featurizer = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(max_features=6000), "title"),
        ("author", TfidfVectorizer(max_features=6000), "author"),
        ("publisher", TfidfVectorizer(max_features=6000), "publisher"),
        ("abstract", TfidfVectorizer(max_features=12000), "abstract"),
        ("ENTRYTYPE", OneHotEncoder(), ["ENTRYTYPE"])
    ],
    remainder='drop'
)
# Transform features
X_train = featurizer.fit_transform(train.drop(columns=['year'])).toarray()
X_val = featurizer.transform(val.drop(columns=['year'])).toarray()
y_train = train['year'].astype(float).values
y_val = val['year'].astype(float).values
X_test = featurizer.transform(test).toarray()

# Transform the 'year' target variable using square transformation
train['year_transformed'] = train['year'] ** 2
val['year_transformed'] = val['year'] ** 2

# Normalize the transformed target variable
y_train_transformed = train['year_transformed'].astype(float).values
y_val_transformed = val['year_transformed'].astype(float).values
y_mean_transformed = y_train_transformed.mean()
y_std_transformed = y_train_transformed.std()

y_train_normalized = (y_train_transformed - y_mean_transformed) / y_std_transformed
y_val_normalized = (y_val_transformed - y_mean_transformed) / y_std_transformed

# Build model function with robust loss function and regularization
def build_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss="mean_absolute_error")
    return model

#scheduler
def scheduler(epoch, lr):
  if epoch < 3:
    return lr
  else:
    return lr * tf.math.exp(-0.2)

# Model training with callbacks
model = build_model(X_train.shape[1])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),tf.keras.callbacks.LearningRateScheduler(scheduler)
history = model.fit(X_train, y_train_normalized, validation_split=0.25, 
          epochs=15, batch_size=32, callbacks=[callback])


# Plotting the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('train_val_loss.png')
plt.show()

# Predict and evaluate
y_val_pred_normalized = model.predict(X_val).flatten()

# Inverse the normalization and apply square root transformation
y_val_pred_transformed = ((y_val_pred_normalized * y_std_transformed + y_mean_transformed))
y_val_pred =  np.sqrt(y_val_pred_transformed)

mae = mean_absolute_error(y_val, y_val_pred)
print(f"Mean Absolute Error on Validation Set: {mae}")

# Predictions for the test set
predictions_normalized = model.predict(X_test).flatten()
predictions = predictions_normalized * y_std + y_mean

# Generate predicted JSON
test['year'] = predictions.round()
test.to_json("predicted.json", orient='records', indent=2)