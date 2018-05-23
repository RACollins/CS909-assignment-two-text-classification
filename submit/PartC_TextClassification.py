#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.metrics import *
from sklearn.metrics.scorer import make_scorer

# define regular expressions, tokeniser, and lemmatiser
lemmatiser = WordNetLemmatizer()
regExLessThan = re.compile(r"&lt;")
regExNonAlpha = re.compile(r"([^\s\w]|_)+")
regExNumeric = re.compile(r"\b\d+\b")

# define functions to process news reports
def replace_less_than(doc):
    processed_doc = regExLessThan.sub("<", doc)
    return(processed_doc)

def remove_non_alpha(doc):
    processed_doc = regExNonAlpha.sub("", doc)
    return(processed_doc)

def remove_extra_whitespace(doc):
    processed_doc = " ".join(doc.split())
    return(processed_doc)

def replace_numeric(doc):
    processed_doc = regExNumeric.sub(" numeric ", doc)
    return(processed_doc)

def lemmatise_doc(doc):
    tokenised_doc = word_tokenize(doc.lower())
    lemmatised_doc = [lemmatiser.lemmatize(token) for token in tokenised_doc]
    processed_doc = " ".join(lemmatised_doc)
    return(processed_doc)

# get the term frequency-inverse-document-frequency of the reports
vectorizer = TfidfVectorizer(stop_words = "english")
def getUnigrams(vec, training_corpus = None, testing_corpus = None):
    if training_corpus:
        X = vec.fit_transform(training_corpus)
        freq_array = X.todense()
        vocab_dict = vectorizer.vocabulary_
    elif testing_corpus:
        X = vec.transform(testing_corpus)
        freq_array = X.todense()
        vocab_dict = vectorizer.vocabulary_
    return(vocab_dict, freq_array)



# define columns of interest
columns_of_interest = ["pid", "fileName", "purpose", "topic.earn", "topic.acq", "topic.money.fx",
"topic.grain", "topic.crude", "topic.trade", "topic.interest",
"topic.ship", "topic.wheat", "topic.corn", "doc.title", "doc.text"]

topic_columns = ["topic.earn", "topic.acq", "topic.money.fx",
"topic.grain", "topic.crude", "topic.trade", "topic.interest",
"topic.ship", "topic.wheat", "topic.corn"]

# open and read in dataset to a pandas dataframe
reports_df = pd.read_csv("reutersCSV.csv", encoding = "ISO-8859-1")
# reduce number of columns
reports_df = reports_df[columns_of_interest]
# drop NaN values
reports_df = reports_df.dropna()
# drop reports whose topic is not listed above
reports_df = reports_df[(reports_df[topic_columns].T != 0).any()]

def get_feature_space(df, training = True):
    processed_reports, y, idVec = [], [], []

    # get id vector
    idVector = df["pid"]

    # process text
    df["doc.text"] = df["doc.text"].apply(replace_less_than)
    df["doc.text"] = df["doc.text"].apply(remove_non_alpha).str.lower()
    df["doc.text"] = df["doc.text"].apply(replace_numeric)
    df["doc.text"] = df["doc.text"].apply(lemmatise_doc)
    df["doc.text"] = df["doc.text"].apply(remove_extra_whitespace)

    unigram_positions, unigram_array = getUnigrams(vectorizer,
    training_corpus = list(df["doc.text"]))

    for index, row in df.iterrows():
        topics = list(row[topic_columns])
        y.append(topics)

    return(idVector, unigram_positions, unigram_array, y)

# create feature space to be used in fitting
print("Generating feature space from training data, please wait...")
IDs, mapping, X, y = get_feature_space(reports_df)


# loop through classifiers a get predictions
scoring = ["accuracy", "precision_micro", "recall_micro", "f1_micro"]
for classifier in ['DecisionTreeClassifier', 'RandomForestClassifier1', 'RandomForestClassifier2']:
    if classifier == 'DecisionTreeClassifier':
        print('Training and testing ' + classifier)
        # generate decision tree classifier
        dtc = tree.DecisionTreeClassifier(max_depth = 11, min_samples_leaf = 1)
        scores = cross_validate(dtc, X, np.array(y), scoring = scoring,
                        cv = 10, return_train_score = False)
        print("Training done!")
        average_scores = {k:np.mean(v) for k, v in scores.items()}
        for k, v in average_scores.items():
            print("{0} : {1:.3f}".format(k, v))

    elif classifier == 'RandomForestClassifier1':
        print('Training and testing ' + classifier)

        # generate random forest classifier
        rfc = RandomForestClassifier(n_estimators = 3, max_depth = None)
        scores = cross_validate(rfc, X, np.array(y), scoring = scoring,
                        cv = 10, return_train_score = False)
        print("Training done!")
        average_scores = {k:np.mean(v) for k, v in scores.items()}
        for k, v in average_scores.items():
            print("{0} : {1:.3f}".format(k, v))

    elif classifier == 'RandomForestClassifier2':
        print('Training and testing ' + classifier)

        # generate random forest classifier
        rfc = RandomForestClassifier(n_estimators = 30, max_depth = None, min_samples_leaf = 2)
        scores = cross_validate(rfc, X, np.array(y), scoring = scoring,
                        cv = 10, return_train_score = False)
        print("Training done!")
        average_scores = {k:np.mean(v) for k, v in scores.items()}
        for k, v in average_scores.items():
            print("{0} : {1:.3f}".format(k, v))
