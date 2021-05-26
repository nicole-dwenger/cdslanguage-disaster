# Classification of Disaster Tweets: The Sun is on Fire!

[Description](#description) | [Methods](#methods) | [Repository Structure](#repository-structure) | [Usage](#usage) | [Results and Disucssion](#results-and-discussion) | [Contact](#contact)


## Description
> This is the self-assigned project of the course Language Analytics. 

This is the self-assigned project. Twitter has become an important platform to monitor and detect emergency situations. However, tweets about disasters can be quite similar to those which share similar words, but have a different meaning. *The sun is on fire!* could refer to a beautiful sunset, while *The church is on fire!* would refer to an emergency. Thus, it is relevant to be able to distinguish between tweets which refer to a true disaster, and those which do not.
The aim of this project was to use two different methods to solve this binary classification task of classifying tweets to refer to a true disaster or not: (1) using tfidf-feature extraction and a logistic regression classifier and (2) using transfer learning by fine-tuning the pre-trained transformer model Bert. Thus, this repository contains two scripts, one for each method, which preprocesses the data, trains and evaluates each of the classifiers. 


## Methods 

### Data and Preprocessing
The data, which was used for this project was downloaded from [Kaggle](https://www.kaggle.com/c/nlp-getting-started/data), where it was published for a Kaggle competition. Only the train.csv  file was used, as the test.csv file did not contain any labels. This data contained 7613 tweets, 4341 tweets tagged as 0 (no disaster) and 3271 tagged with 1 (disaster). For this project, 3000 tweets of each label (0,1) were sampled, leading to a total of 6000 tweets. For both, the logistic regression classifier and Bert, the tweets were preprocessed using the following steps: 

1. Remove emojis. They also have been replaced with their descriptions, but I chose to remove them to only focus on the text. However, they could potentially also help to distinguish between disaster/no disaster. 
2. Remove digits and their appendencies, e.g. 40ms
3. Remove mentiones, i.e. @...
4. Remove URLs

**Only** for the logistic regression classifier the following steps were also performed (they were not performed for Bert, since it seemed relevant to keep all context words and not remove any words):

5. Remove punctuation
6. Lower tokens
7. Lemmatise tokens
8. Filter out stop words usig NLTKs english stopwords. 

For both models, the data was split into train and test data using an 70/30 split. 

### Tfidf Vectorisation and Logistic Regression
Tfidf (term frequency-inverse document frequency) is a measure of how often a given word occurs in a document, while also taking into account how often it occurs across documents. Thus, after preprocessing, the tweets were transformed into feature spaces using tfidf-vectorisation, while keeping a maximum of 500 features. Subsequently, the logistic regression classifier was trained using default parameters, and evaluated based on predictions of the model for the test data.  

### Fine-turning Bert
While tfidf-vectorisation may in some cases be sufficient to classify text data, it has several advantages, as it only focuses on the occurrence of single words. Consequently, it cannot take into account the relation between words, and the temporal development of text. In other words, it cannot take into account the local or global context in which they appear. Bert is a complex transformer model, which can take into account more contextual relations, as it makes use of attention, meaning it also considers which relate to the word which is currently processed. The model has been trained on English Wikipedia and a BookCorpus (a total of 3300 million words!). This complex model can be used in various ways in transfer learning, one of which one is fine-tuning. This means, that to the transformer output of the model, a classification layer is added and trained to classify the given data. 
To use fine-tune this model for the classification of tweets, it was necessary to tokenise and encode it, which was done using the pre-trained Bert tokeniser. In this step, all tweets were tokenised, special tokens of the Bert model (e.g. [SEP], [CLS]) were added, and the vectors were padded to be have maximum length of 150 (the longest tweet was 137 after preprocessing). Lastly, an attention mask was added, to make sure the model does not focus on the padded elements. The encoded test and train data were used to fine-tune the pre-trained Bert-Based-Uncased model, to classify the data. Adam with a learning rate of 0.00002 was used as optimiser and sparse categorical cross-entropy as loss function. The model was trained for 2 epochs and evaluated based on predictions on the test data. This process could probably have been improved by e.g. adding more layers or adjusting parameters, however for simplicity and learning, I tried to follow class content and this [tutorial](https://atheros.ai/blog/text-classification-with-transformers-in-tensorflow-2).


## Repository Structure 
```
|-- data/
    |-- train.csv
    
|-- out/                                # Directory for output, corresponding to scripts
    |-- LR/                             # Directory for output of logistic regression classifier using tfidf
        |-- LR_report.txt               # Classification report
        |-- LR_matrix.png               # Classification/confusion matrix
    |-- BERT/                           # Directory for output of classification using Bert
        |-- BERT_history.png            # Training history of model
        |-- BERT_report.txt             # Classification report
        |-- BERT_matrix.png             # Classification/confusion matrix

|-- src/                                # Directory containing main scripts of the project
    |-- TDIF_disaster.py                # Script for disaster classification using logistic regression, tfidf
    |-- BERT_disaster.py                # Script for disaster classification using Bert
   
    
|-- README.md
|-- create_venv.sh                       # Bash script to create virtual environment
|-- requirements.txt                     # Dependencies, installed in virtual environment

```

## Usage 
**!** The script for the logistic regression classifier (LR_disaster.py) has only been tested on Linux, using Python 3.6.9 (worker02), while the script using Bert has only been tested on Linux using Python 3.7 (UCloud). 

### 1. Cloning the Repository and Installing Dependencies
To run the scripts in this repository, I recommend cloning this repository and installing necessary dependencies in a virtual environment. The bash script `create_venv.sh` can be used to create a virtual environment called `venv_disaster` with all necessary dependencies, listed in the `requirements.txt` file. The following commands can be used:

```bash
# cloning the repository
git clone https://github.com/nicole-dwenger/cdslanguage-disaster.git

# move into directory
cd cdslanguage-disaster/

# install virtual environment
bash create_venv.sh

# activate virtual environment 
source venv_disaster/bin/activate
```

### 2. Data and Pretrained Embeddings
The raw data, which was downloaded from [Kaggle](https://www.kaggle.com/c/nlp-getting-started/data) is stored in the `data/` directory. Thus, after cloning the repository it is not necessary to retrieve any additional data to run the scripts. 

### 3.1. Script for Tfidf-Vectorisation and Logistic Regression: LR_disaster.py
The script `LR_disaster.py` preproceses the tweets, following the steps outlined above, extracts features using Tfidf, trains and evaluates a logistic regression classifier. The script should be called from the `src/` directory:

```bash
# move into src
cd src/

# run script with default parameters
python3 LR_disaster.py

# run script with specified parameters
python3 LR_disaster.py -o out_lr
```
__Parameters:__

- `-i, --input_file`: *str, optional, default:* `../data/train.csv`\
  Path to input file (raw disaster data). 
  
- `-o, --output_name`: *str, optional, default:* `LR`\
  Name of output directory, in which output is saved. The directory with the given name will be created in `out/`. 
    
__Output__ saved in `out/{output_name}/`: 

- `LR_report.txt`\
   Classification report of the logistic regression classifier.
   
- `LR_matrix.png`\
   Classification/confusion matrix of the logistic regression classifier. 
   
### 3.2. Classification using fine-tuning of Bert: BERT_disaster.py
The script `BERT_disaster.py` preproceses the tweets, following the steps outlined above, encodes the tweets, trains and evaluates the fine-tuned Bert model. The script should be called from the `src/` directory:

```bash
# move into src
cd src/

# run script with default parameters
python3 BERT_disaster.py

# run script with specified parameters
python3 BERT_disaster.py -e 2 -s 200
```
__Parameters:__

- `-i, --input_file`: *str, optional, default:* `../data/train.csv`\
  Path to input file (raw disaster data). 
  
- `-s, --data_sample`: *int, optional, default:* `6000`\
  Total number of tweets to sample. They will be sampled equally for each target label, i.e. if `6000` this means 3000 with label 0 and 3000 with label 1. For testing the script, this should be decreased!
  
- `-r, --epochs`: *int, optional, default:* `2`\
  Number of epochs to train the model. 
  
- `-o, --output_name`: *str, optional, default:* `BERT`\
  Name of output directory, in which output is saved. The directory with the given name will be created in `out/`. 
    
__Output__ saved in `out/{output_name}/`: 

- `BERT_history.png`\
   Visualisation of training history over epochs.

- `BERT_report.txt`\
   Classification report of the model. 
   
- `BERT_matrix.png`\
   Classification/confusion matrix of the model.
   
 
## Results and Discussion 
All results can be found in the `out/` directory of the GitHub repository. The classification report of the [logistic regresison classifier](https://github.com/nicole-dwenger/cdslanguage-disaster/blob/master/out/LR/LR_report.png) indicated an overall weighted F1 score of 0.77. Specifically, it achieved an F1 score of 0.78 for classification of no-disaster-tweets (0) and an F1 score of 0.76 for classification of disaster-tweets (1). Considering the simplicity of the method, this seems relatively good.  The fine-tuned [Bert model](https://github.com/nicole-dwenger/cdslanguage-disaster/blob/master/out/BERT/BERT_report.png) achieved an overall weighted F1 score of 0.82, and an F1 score of 0.82 for both no-disaster tweets (0) and disaster-tweets (1). This means an increase of .05 for Bert compared to the baseline logistic regression classifier. The model history indicates, that the model was learning faster on the training data, while it did not improve much on the testing data over epochs: 

<p align="center">
  <img width="350" src="https://github.com/nicole-dwenger/cdslanguage-disaster/blob/master/out/BERT/BERT_history.png">
</p>

Thus, more epochs might have caused overfitting. However, it should be mentioned that in this project the baseline Bert model was used. Additional layers and tweaks to parameters might further improve the model. Similarly, some of the preprocessing steps could have been adjusted, to further clean the tweets and remove non-content characters. Nevertheless, it was a great first encounter with Bert. Thank you for introducing us, Ross!


## Contact
If you have any questions, feel free to contact me at 201805351@post.au.dk. 