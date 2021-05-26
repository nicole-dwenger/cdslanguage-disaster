#!/usr/bin/python

"""
Classify disaster tweets (0=no distaster, 1=disaster) by fine-tuning pretrained Bert.

Steps:
  - Load data, only keep relevant columns
  - Balance data: sample tweets for disaster and no disaster equally
  - Clean tweets: 
      - remove emojis, digits with appendicies, mentiones (@..), urls
  - Split into train and test with 70/30 split
  - Prepare and encode data for Bert using tokenizer from Bert
  - Fine-tune pretrained Bert on disaster data, save training history
  - Generate predictions to get classification report and classification matrix
  
Input: 
  - -i, --input_file, str, optional, default: ../data/train.csv, path to input file (raw disaster tweets)
  - -e, --epochs, int, optional, default: 2, number of epochs to trian the model for
  - -s, --data_sample, int, optional, default: 6000, how many tweets should be used in total (0 and 1 combined)
  - -o, --output_name, str, optional, default: BERT, name of output directory, created in out/
  
Output saved in out/{output_name}:
  - BERT_history.png: visualisation of training history
  - BERT_report.txt: classification report of classifier
  - BERT_matrix.png: classification matrix of classifier
"""

# DEPENDENCIES ---------------------------------------

# Basics
import os
import re
import string
import numpy as np
import pandas as pd
import argparse

# Preprocessing
import demoji
demoji.download_codes()

# Tensorflow 
# Hide warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Transformers and BERT
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Sklearn for output
from sklearn.metrics import classification_report

# Utils
import sys
sys.path.append(os.path.join(".."))
from utils.disaster_utils import (classification_matrix, save_model_history, 
                                  save_model_report, save_model_matrix)

# HELPER FUNCTIONS ---------------------------------------

def clean_tweets(tweet):
    """
    Clean tweets by removing emojis, removing everything in brackets, 
    removing digits with appicies, remove mentions and urls.
    Input:
      - tweet: str of tweet
    Returns:
      - out_tweets: str, cleaned tweet
    """
    # remove emojis
    tweet = demoji.replace(tweet, "")
    # remove anything that is in brackets, e.g. time
    tweet = re.sub(r"[\[].*?[\]]", "", tweet)
    # remove digtis with letters before
    tweet = re.sub(r"\w+\d+", "", tweet)
    # remove digits with letters after
    tweet = re.sub(r"\d+\w+", "", tweet)
    # remove mentions
    tweet = re.sub(r'@[\w]+ ','', tweet)
    # remove urls
    out_tweet = re.sub(r'http\S+', '', tweet)
    
    return out_tweet

def split_data(data, train_size=0.7):
    """
    Split data into training and test, the manual way
    Input: 
      - data: dataframe with labels and cleaned tweets
    Returns:
      - two df: train and test
    """
    # Drop text column, to only keep the cleaned ones
    data = data.drop(columns = ["text"])
    # Shuffle data
    shuffle_data = data.sample(frac=1)
    # Define number of trainign samples based on size
    train_size = int(train_size * len(data))
    # Select and test and train data
    train = shuffle_data[:train_size]
    test = shuffle_data[train_size:]
    
    return train, test

def tokenize(text):
    """
    Tokenize texts using BERT tokenizer: 
      - Add special tokens, pad to max leength, and add attention mask
    Input: 
      - text documnet
    Returns:
      - tokenized text
    """
    return tokenizer.encode_plus(
            text,
            # Add special tokens for Bert
            add_special_tokens=True,
            # Define max length (after cleaning max length was 137
            max_length=150,
            # Allow trunctuation
            truncation=True,
            # Otherwise pad to the defined max length
            padding="max_length",
            # Return tokens ids
            return_token_type_ids=True,
            # Add attention mask to not focus on paddig
            return_attention_mask=True)

def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    """
    Map input ids and tokens to expected output for Bert
    - Function adjusted from class
    """
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,}, label

def encode_examples(df, limit=-1):
    """
    Summary function, to turn texts into format for transformer Bert
    - Function adjusted from class
    """
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    if (limit > 0):
        df = df.take(limit)

    for label, text in np.array(df):
        bert_input = tokenize(str(text))
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
        
    out = tf.data.Dataset.from_tensor_slices((input_ids_list, 
                                              attention_mask_list, 
                                              token_type_ids_list, 
                                              label_list)).map(map_example_to_dict)
    
    return out

def compile_bert(learning_rate):
    """
    Load pretrained bert based uncased model, append optimizer, loss and metric, 
    and finally compile model to be trained
    Input: 
      - learning_rate: rate to learn for model
    Returns:
      - model: compiled model
    """

    # Load pretrained BERT model
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

    # Use Adam as optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, 
                                         epsilon=1e-08)

    # Use sparce categorical cross entropy and accuracy
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    # Comile the model
    model.compile(optimizer=optimizer, 
                  loss=loss, 
                  metrics=[metric])
    
    return model

def get_prediction_labels(model, test_encoded):
    """
    Generate predictions and turn output from transformer model back into 
    labels, so that they can be compared to the actual true labels
    Input: 
      - model: trained model
      - test_encoded: encoded test data
    Reuturns: 
      - array of predicted test labels (0 or 1)
    """
    # Generate predictions
    predictions = model.predict(test_encoded, batch_size=32)
    # Extract logits from predictions
    logits = predictions["logits"]
    # Get max to get true labels
    pred_labels = np.argmax(logits, axis = 1)

    return pred_labels


# MAIN FUNCTION ------------------------------------------

def main():
    
    # --- ARGUMENT PARSER ---
    
    # Initialise argument parser
    ap = argparse.ArgumentParser()
    
    # Input option for input file 
    ap.add_argument("-i", "--input_file", type=str, help="Path to input file", 
                    required=False, default = "../data/train.csv")
    
    # Input option for sample
    ap.add_argument("-s", "--data_sample", type=int, help="To reduce processing time: n of tweets to use in total", 
                    required=False)
    
    # Input option for epochs
    ap.add_argument("-e", "--epochs", type=int, help="Number of epochs to train the model for", 
                    required=False, default=5)
    
    ap.add_argument("-o", "--output_name", type=str, help="Name of output directory, will be created in out/", 
                    required=False, default="BERT")
    
    # Retrieve arguments
    args = vars(ap.parse_args())
    input_file = args["input_file"]
    data_sample = args["data_sample"]
    epochs = args["epochs"]
    output_name = args["output_name"]
    
    # --- PREPARE DATA ---

    # Load data
    data = pd.read_csv(input_file).drop(columns = ["id", "keyword", "location"])
    # Sample data (half of total sample for each target label)
    data = data.groupby("target").sample(int(data_sample/2), random_state=1)
    
    # Clean Tweets
    data["cleaned"] = data.apply(lambda row : clean_tweets(row['text']), axis=1)
    
    # Select and split
    train, test = split_data(data, train_size=0.7)
    
    # Encode data and batch 
    train_encoded = encode_examples(train).shuffle(1000).batch(32)
    test_encoded = encode_examples(test).batch(32)
    
    # --- MODEL ---
    
    # Compile model
    model = compile_bert(learning_rate=2e-5)
    
    # Train model
    history = model.fit(train_encoded, 
                         epochs=epochs,
                         batch_size=32,
                         validation_data=test_encoded)
    
    # Evaluate model: get predictions
    predictions = get_prediction_labels(model, test_encoded)
    # Generate classification report
    report = classification_report(test["target"].values, predictions)
    # Generate classification matrix
    matrix = classification_matrix(test["target"].values, predictions, model_name="BERT")
    
    # Print classification report to command line
    print(f"\nClassification report:\n{report}")
    
    # -- OUTPUT ---
    
    # Prepare output directory
    output_directory = os.path.join("..", "out", output_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    # Save model history
    save_model_history(history, epochs, output_directory, "BERT_history.png")
    # Save classification report
    save_model_report(report, output_directory, "BERT_report.txt")
    # Save model matrix
    save_model_matrix(matrix, output_directory, "BERT_matrix.png")
    
    
if __name__=="__main__":
    main()