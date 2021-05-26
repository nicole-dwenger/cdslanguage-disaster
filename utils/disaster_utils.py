#!/usr/bin/python

"""
Utility functions for classification of disaster tweets using Logistic Regression or BERT. 

Functions: 
  - classification_matrix: create classification matrix from actual and predicted labels
  - save_model_history: save visualisation of training history (only for BERT)
  - save_model_report: save classifiaction report of model
  - save_model_matrix: save classification matrix of model
"""

# DEPENDENCIES -------------------------------------

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# FUNCTIONS -------------------------------------

def classification_matrix(actual, predictions, model_name):
    """
    Function to plot classification matrix
    Input:
      - actual: array of actual label names
      - predictions: array of predicted label names
    Returns:
      - classification_matrix
    """
    # Create confusion matrix
    cm = pd.crosstab(actual, predictions, rownames=['Actual'], 
                     colnames=['Predicted'], normalize='index')
    
    # Initialise figure
    p = plt.figure(figsize=(10,10))
    # Addd title
    p = plt.title(f"Classification Matrix for Classification of Disaster Tweets using {model_name} (0 = no disaster, 1 = disaster)")
    # Plot confusion matrix on figure as heatmap
    p = sns.heatmap(cm, annot=True, fmt=".2f", cbar=False)
    
    # Save the figure in variable
    classification_matrix = p.get_figure()
        
    return classification_matrix

def save_model_history(history, epochs, output_directory, filename):
    """
    Plotting the model history, i.e. loss/accuracy of the model during training
    Input: 
      - history: model history
      - epochs: number of epochs the model was trained on 
      - output_directory: desired output directory
      - filename: name of file to save history in
    """
    # Define output path
    out_history = os.path.join(output_directory, filename)

    # Visualize history
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_history)
    
def save_model_report(report, output_directory, filename):
    """
    Save report to output directory
    Input: 
      - report: model classifcation report
      - output_directory: final output_directory
      - filename: name of file to save report in
    """
    # Define output path and file for report
    report_out = os.path.join(output_directory, filename)
    # Save report in defined path
    with open(report_out, 'w', encoding='utf-8') as file:
        file.writelines("Classification report:\n")
        file.writelines(report) 

def save_model_matrix(matrix, output_directory, filename):
    """
    Save model matrix in outptut directory
    Input:
      - matrix: plot of classification matrix
      - output_directory: path to output directory
      - filename: desired filename
    """
    # Define output path 
    out_matrix = os.path.join(output_directory, filename)
    # Save matrix
    matrix.savefig(out_matrix)
    
if __name__=="__main__":
    pass