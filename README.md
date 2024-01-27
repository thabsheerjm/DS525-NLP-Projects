# DS525-NLP-Projects
The repository contains three assignments done towards the completion of the course 'Natural Language Processing'. 

## Assignment 1 : Fake News Detection 

This project aims to analyze and classify news articles into 'Real' and 'Fake' categories using Natural Language Processing (NLP) and Machine Learning techniques. The dataset, sourced from Kaggle, contains separate collections of real and fake news articles. The primary objective is to build a model that can accurately distinguish between real and fake news based on the content of the articles.

The following image shows the word frequency in the dataset as a word cloud.   

<img src="Assignment_1/dep/d1.png" align="center" alt="undistored" width="400">

### Data Processing and Exploration

#### 1. Data Loading and Cleaning:

  * The dataset is initially segregated into 'real' and 'fake' news files.
  * Basic exploratory data analysis is conducted to understand the composition and structure of the data.
  * Necessary preprocessing steps include removing duplicates and handling any null values.

#### 2. Data Visualization:
  * Visual representations (like pie charts) are used to understand the distribution of real vs. fake news.
  * Word clouds are generated to visualize the most frequent words in both categories.
  * Histograms are used to compare the length of articles in terms of characters and words between real and fake news.

#### 3. Data Preprocessing for NLP:
  * Text data is cleaned by removing special characters, unnecessary spaces, and HTML tags.
  * Text normalization techniques like stemming and lemmatization are applied.
  * Common stopwords are removed to focus on more meaningful words in the text.

### Feature Engineering

#### 1. Creation of a Corpus:
  * A combined text column is created by merging titles with the main text content.
  * This text is further processed to create a corpus for NLP modeling.

#### 2. POS Tagging
 *  Part-of-Speech (POS) tagging is experimented with to include only nouns and adjectives in the text, aiming to enhance the model's focus on significant words.

### 3. Vectorization: 
  * The text data is converted into numerical form using techniques like CountVectorizer and TF-IDF (Term Frequency-Inverse Document Frequency) for model training.
  
### Model Building and Evaluation
#### 1. Model Training:
  * Different machine learning models like Multinomial Naive Bayes and Logistic Regression are trained on both the Bag of Words model and TF-IDF features.
#### 2. Model Evaluation:
  * Models are evaluated based on accuracy, precision, recall, and F1-score.
  * Confusion matrices are plotted to understand the true positives, true negatives, false positives, and false negatives for each model.

### Results 
  * The Naive Bayes model showed a higher accuracy with the Bag of Words approach compared to the TF-IDF method.
  * Logistic Regression also performed well, with detailed classification reports providing insights into the precision and recall of the models.
  * The project demonstrates the effectiveness of using NLP techniques and machine learning for classifying news articles into real and fake categories.
  * The choice of text preprocessing and feature engineering techniques, like POS tagging and vectorization methods, significantly influences model performance.

For more details see the [Report](https://github.com/thabsheerjm/DS525-NLP-Projects/blob/main/Assignment_1/dep/Report.pdf) 

### How to Run
In the directory, run the notebook `fake_news_detection.ipynb`  

## Assignment 2 : Sentiment Analysis via Text Classification

<img src="Assignment_2/dep/d2.png" align="center" alt="undistored" width="400">

This project leverages deep learning techniques, particularly the use of the BERT (Bidirectional Encoder Representations from Transformers) model, for sentiment analysis. Sentiment analysis is a subfield of NLP (Natural Language Processing) that involves determining the emotional tone behind a body of text. This is a crucial task in various applications like understanding customer sentiments in reviews, social media monitoring, brand monitoring, and more.

### 1. Data preprocessing: 
  * The dataset, which likely contains text data labeled with sentiments, is loaded and preprocessed. This step usually involves cleaning the text data, handling missing values, and possibly encoding the labels (sentiments).

### 2. BERT Tokenizaton:
  * The BERT tokenizer from the transformers library is initialized. This tokenizer converts text data into a format (like token IDs, attention masks) that is compatible with the BERT model.

### 3. Dataset and DataLoader Configuration:
  * Custom dataset classes are defined to handle the training and validation datasets.
  * PyTorch **DataLoader** objects are configured for efficient data batching and loading during the model training and validation phases.

### Model Training:
  * A function for fine-tuning the BERT model on the sentiment analysis task is defined.
  * The model is trained with the specified training dataset, using a loss function and optimizer. The training process typically involves adjusting the weights of the neural network on the training data to minimize the loss function.

### Model Saving and Loading:
 * The trained model is saved to a file for later use.
 * The notebook demonstrates loading the trained model for evaluation purposes.

### Testing and Evaluation:
  * A dataset for validation or testing is prepared.
  * The model’s performance is evaluated on this dataset. This might involve computing metrics like accuracy, precision, recall, and F1-score to gauge how well the model performs in classifying the sentiments.

For Results see the [Report](https://github.com/thabsheerjm/DS525-NLP-Projects/blob/main/Assignment_2/dep/Report.pdf)

### How to Run

In the directory, run the notebook `sentiment_analysis.ipynb` 

## Assignment 3 : Question Answering via Reading Comprehension

 The project's goal is to create a model that can accurately answer questions based on given contexts, a common task in NLP known as question answering or reading comprehension.
 
 <img src="Assignment_3/dep/d3.png" align="center" alt="undistored" width="400">
 
### 1. Data Loading:

  * The project uses the SQuAD (Stanford Question Answering Dataset) for training and validation, which is a popular dataset in NLP for question-answering tasks.

### 2. Data Exploration:

  * The notebook includes analysis of the number of questions and unique contexts in both training and validation datasets.

### 3. Data Preparation for Model:

  * The contexts, questions, and answers are extracted from the dataframes and prepared for model training and validation.
  * A function add_end_idx is likely used to add the ending index of answers in the context, a necessary step for the model to understand where answers end in the given text.

### 4. Model Training and Evaluation:

  * The notebook details the process of fine-tuning the BERT model for the specific task of reading comprehension.
  * The model is trained using the training dataset and then saved for future use.
  * Evaluation of the model is performed using the validation dataset, focusing on metrics like accuracy and possibly F1-score.

 For results and discussions see the [Report](https://github.com/thabsheerjm/DS525-NLP-Projects/blob/main/Assignment_3/dep/Report.pdf)

 ### How to Run

In the directory, run the notebook `reading_comprehension.ipynb` 
 
## Acknowledgement 

The contents of this repository are the assigments done towards the completion of the course DS525 - Natural Language Processing at Worcester Polytechnic Institute. 
