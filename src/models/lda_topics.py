"""
!Note test done only for papers in english 
Used following videos as reference for the baseline model:
https://www.youtube.com/playlist?list=PL2VXyKi-KpYttggRATQVmgFcQst3z6OlX
github repository:
https://github.com/wjbmattingly/topic_modeling_textbook/blob/main/03_03_lda_model_demo_bigrams_trigrams.ipynb
"""
import numpy as np
import json
import glob
import pandas as pd

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy
import spacy
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

# vis
import pyLDAvis
import pyLDAvis.gensim

stopwords = stopwords.words("english")

# Load datasets with the language detected currently available in data_raw branch:
train_lda = "poc_feature_train_eda_j.json"
test_lda = "poc_feature_test_eda_j.json"
train_lda_df = pd.read_json(train_lda)
test_lda_df = pd.read_json(test_lda)
# train_lda_df.shape

# Test example
data_1 = [
    "Studying at tilburg University can provide you a thorough understunding of society as well gain a title on a specific field based on your studies"
]

# Baseline model done using english records from training dataset.
sample_data = train_lda_df[train_lda_df["lang_tit_publ_low"] == "en"]
data = sample_data["title_low"].to_list()  # LDA Model requires a list as an input
len(data)


# nlp lematization from the title_low feature (Raw function, finetune may be needed?)
def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for text in texts:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return texts_out


lemmatized_texts = lemmatization(data)
print(lemmatized_texts[0])


def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return final


data_words = gen_words(lemmatized_texts)
print(data_words)

"""
Bigrams and trigras not review resutls perhaps may be interesting to finetune it 
"""
# BIGRAMS AND TRIGRAMS
bigram_phrases = gensim.models.Phrases(data_words, min_count=4, threshold=100)
trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=100)

bigram = gensim.models.phrases.Phraser(bigram_phrases)
trigram = gensim.models.phrases.Phraser(trigram_phrases)


def make_bigrams(texts):
    return [bigram[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram[bigram[doc]] for doc in texts]


data_bigrams = make_bigrams(data_words)
data_bigrams_trigrams = make_trigrams(data_bigrams)

print(data_bigrams_trigrams[5])


"""
Important step to remove not relevant most frequent words
"""
# TF-IDF REMOVAL
from gensim.models import TfidfModel

id2word = corpora.Dictionary(data_bigrams_trigrams)

texts = data_bigrams_trigrams

corpus = [id2word.doc2bow(text) for text in texts]
# print (corpus[0][0:20])

tfidf = TfidfModel(corpus, id2word=id2word)

low_value = 0.03
words = []
words_missing_in_tfidf = []
for i in range(0, len(corpus)):
    bow = corpus[i]
    low_value_words = []  # reinitialize to be safe. You can skip this.
    tfidf_ids = [id for id, value in tfidf[bow]]
    bow_ids = [id for id, value in bow]
    low_value_words = [id for id, value in tfidf[bow] if value < low_value]
    drops = low_value_words + words_missing_in_tfidf
    for item in drops:
        words.append(id2word[item])
    words_missing_in_tfidf = [
        id for id in bow_ids if id not in tfidf_ids
    ]  # The words with tf-idf socre 0 will be missing

    new_bow = [
        b
        for b in bow
        if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf
    ]
    corpus[i] = new_bow

"""
This step needs to be further finetune and optimize 
"""
# LDA Baseline model
lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus[:-1],
    id2word=id2word,
    num_topics=10,
    random_state=100,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha="auto",
)

# lda_model.save("train_model.model")
