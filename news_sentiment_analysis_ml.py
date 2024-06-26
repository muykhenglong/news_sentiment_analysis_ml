#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:59:26 2024

@author: s.o
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Import dataset
data = pd.read_csv('news_df.csv')
data.drop(['Unnamed: 0', 'Date'], axis=1, inplace=True)

# Extract features and labels
X = data.iloc[:, 1]
y = data.iloc[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Tokenize the news text and convert data to matrix format
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)

# Transform data by applying term frequency inverse document frequency (TF-IDF)
tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train_vec).toarray()

# Train the NB classifier
clf = GaussianNB().fit(X_train_tfidf, y_train)

# Process test dataset
# Tokenize the news text and convert data to matrix format
X_test_vec = vectorizer.transform(X_test)

# Transform data by applying term frequency inverse document frequency (TF-IDF)
X_test_tfidf = tfidf.transform(X_test_vec).toarray()

# Predict the sentiment values
y_pred = clf.predict(X_test_tfidf)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Print detailed classification report
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

# Cross-validation
cv_scores = cross_val_score(clf, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean():.4f}")


"""
Note: 
The model requires further work to improve accuracy including 
- maticulously extracting and labeling the dataset
- preprocessing data (apply stemming or lemmatization)
- feature engineering (balance class) and hyperparameter tuning 
- model expiriment
"""

