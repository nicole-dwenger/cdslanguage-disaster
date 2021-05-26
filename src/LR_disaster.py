#!/usr/bin/python

"""
Classify disaster tweets (0=no distaster, 1=disaster) using Tfidf-Vectorisation and Logistic Regression.

Steps:
  - Load data, only keep relevant columns
  - Balance data: 3000 tweets for disaster and no disaster = total of 6000
  - Clean tweets: 
      - remove emojis, digits with appendicies, mentiones (@..), urls, punctuation
      - lower and lemmatise tokens, remove stopwords
  - Split into train and test with 70/30 split
  - Extract tfidf features of train and test tweets
  - Train logistic regression classifier
  - Generate classification report and classification matrix, save in output 
  
Input: 
  - -i, --input_file, str, optional, default: ../data/train.csv, path to input file (raw disaster tweets)
  - -o, --output_name, str, optional, default: LR, name of output directory, created in out/
  
Output saved in out/{output_name}:
  - LR_report.txt: classification report of classifier
  - LR_matrix.png: classification matrix of classifier
"""

# LIBRARIES -------------------------------------------

# Basics
import os
import pandas as pd
import numpy as np
import re
import string
import argparse

# Emojis
import demoji
demoji.download_codes() 

# NLTK for stopwords and lemmatisation
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Sklearn for Tdidf and ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Utils
import sys
sys.path.append(os.path.join(".."))
from utils.disaster_utils import (classification_matrix, save_model_report, save_model_matrix)


# HELPER FUNCTIONS ---------------------------------------

def clean_tweets_advanced(tweet):
    """
    Cleaning tweets for tdfif vectorisation:
      - remove non-text content aspects (emoji, brackets, digits, mentions, url)
      - remove punctuation
      - lemmatise and remove stopwords
    Input: 
      - tweet: unprocessed tweet as str
    Returns:
      - tweet_out: cleaned tweet as str
    """
    # Remove emojis
    tweet = demoji.replace(tweet, "")
    # Remove anything that is in brackets, e.g. time
    tweet = re.sub(r"[\[].*?[\]]", "", tweet)
    # Remove digtis with letters before
    tweet = re.sub(r"\w+\d+", "", tweet)
    # Remove digits with letters after
    tweet = re.sub(r"\d+\w+", "", tweet)
    # Remove mentions
    tweet = re.sub(r'@[\w]+ ','', tweet)
    # Remove urls
    tweet = re.sub(r'http\S+', '', tweet)
    # Remove punctuation
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Make lower
    tweet = tweet.lower()
    # Tokenize
    tokens = nltk.word_tokenize(tweet)
    # Lemmatise
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # Join back together
    tweet_out = " ".join(tokens)
    
    return tweet_out

def split_data(data, test_size):
    """
    From dataframe containing columns for the cleaned text ("cleaned") and labels "target",
    split into text (X) and labels (y) into train and test 
    Input: 
      - data: dataframe with text and labels
      - test_size: size of test data
    Returns: 
      - X_train, X_test: train and test texts
      - y_train, y_test: train and test labels
    """
    # Get texts (X) and labels (y)
    X = data["cleaned"].values
    y = data["target"].values
    
    # Split into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    
    return X_train, X_test, y_train, y_test

def extract_tfidf(X_train, X_test, max_features):
    """
    Extract features of train and test texts using Tfidf-Vectoriser 
    Input: 
      - X_train, X_test: array of train and test texts
      - max_features: max features to keep 
    Returns: 
      - X_train_features, X_test_features: extracted features of train and test texts
      - feature_names: extracted tokens
    """
    # Initialise vectoriser
    vectorizer = TfidfVectorizer(ngram_range = (1,2), lowercase= True, max_features=max_features)
    # Apply to train
    X_train_features = vectorizer.fit_transform(X_train)
    # Apply to test
    X_test_features = vectorizer.transform(X_test)
    # Get feature names
    feature_names = vectorizer.get_feature_names()
    # Print how many features were extracted
    print(f"Number of features extracted: {len(feature_names)}")
    
    return X_train_features, X_test_features


# MAIN FUNCTION ------------------------------------------

def main():
    
    # --- ARGUMENT PARSER ---
    
    # Initialise argument parser
    ap = argparse.ArgumentParser()
    
    # Input option for input file 
    ap.add_argument("-i", "--input_file", type=str, help="Path to input file", 
                    required=False, default = "../data/train.csv")
    
    # Input option for output directory name
    ap.add_argument("-o", "--output_name", type=str, help="Name of output directory, will be created in out/", 
                    required=False, default="LR")
    
    # Retrieve arguments
    args = vars(ap.parse_args())
    input_file = args["input_file"]
    output_name = args["output_name"]
    
    # --- PREPARE DATA ---
    
    # Print message
    print("\nInitialising feature extraction using Tdfif and Logistic Regression Classifier.")
    
    # Load data and drop irrelavent colums
    data = pd.read_csv(input_file).drop(columns = ["id", "keyword", "location"])
    # Select 3000 for each target, for balanced data
    data = data.groupby("target").sample(3000, random_state=1)
    
    # Clean tweets 
    data["cleaned"] = data.apply(lambda row : clean_tweets_advanced(row['text']), axis = 1)
    
    # Get test and train texts (X) and labels (y)
    X_train, X_test, y_train, y_test = split_data(data, test_size=0.3)
        
    # --- TFIDF and LOGISTIC REGRESSION ---
  
    # Extract tdif features
    X_train_features, X_test_features = extract_tfidf(X_train, X_test, 500)
    
    # Train logistic regression classifier
    clf = LogisticRegression(random_state=42, max_iter=1000).fit(X_train_features, y_train)
    
    # Evaluate classifier: generate predictions
    predictions = clf.predict(X_test_features)
    # Create classification report
    report = classification_report(y_test, predictions)
    # Create classification matrix
    matrix = classification_matrix(y_test, predictions, model_name="LR")
    
    # Print classification report to command line
    print(f"Classification report:\n{report}")
          
    # --- OUTPUT ---
    
    # Prepare output directory    
    output_directory = os.path.join("..", "out", output_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Save the classification report
    save_model_report(report, output_directory, "LR_report.txt")
    # Save the classification matrix
    save_model_matrix(matrix, output_directory, "LR_matrix.png")
    
    # Print message
    print(f"All done! Output saved in {output_directory}!")

    
if __name__=="__main__":
    main() 