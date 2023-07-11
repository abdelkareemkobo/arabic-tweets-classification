# Arabic Tweet Classification Project

This project focuses on cleaning and classifying a dataset containing Arabic tweets into two categories: spam and ham (non-spam). The classification is performed using the mBERT (multilingual BERT) model, which is a state-of-the-art language model developed by Google. Additionally, a Streamlit demo is provided for easy interaction with the model.

## Table of Contents
- [Arabic Tweet Classification Project](#arabic-tweet-classification-project)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Data Preprocessing](#data-preprocessing)
  - [Training the mBERT Model](#training-the-mbert-model)
  - [Evaluation](#evaluation)
  - [TODO](#todo)
  - [Streamlit Demo](#streamlit-demo)
  - [Contributing](#contributing)
  - [License](#license)

## Introduction

The goal of this project is to develop a machine learning model capable of accurately classifying Arabic tweets as spam or ham. The cleaning and preprocessing steps are crucial to ensure the model's effectiveness. The mBERT model is utilized due to its ability to handle multilingual tasks, including Arabic text classification. The Streamlit demo provides an interactive interface for users to input their own tweets and obtain predictions.

## Installation

To run this project, you need to have the following dependencies installed:

- Python 3
- TensorFlow
- Transformers library (Hugging Face)
- Pandas
- NumPy
- Streamlit

You can install the required libraries by executing the following command:

```
pip install tensorflow transformers pandas numpy streamlit
```

## Data Preprocessing

The dataset used for this project should be in CSV format, with two columns: "text" containing the tweet text, and "label" indicating the class (spam or ham). Prior to training the model, perform the following preprocessing steps:

1. **Data Cleaning**: Remove any irrelevant or unnecessary information, such as URLs, hashtags, and special characters. Additionally, eliminate any duplicates or empty entries.

2. **Tokenization**: Tokenize the cleaned tweet text into individual words or subwords. This step is essential for encoding the text for input to the mBERT model.

3. **Label Encoding**: Convert the categorical labels (spam and ham) into numerical form for training the model. Assign 0 for ham and 1 for spam.

## Training the mBERT Model

1. **Loading the Dataset**: Load the preprocessed dataset into memory using a suitable library like Pandas.

2. **Splitting the Data**: Split the dataset into training, validation, and testing sets. Typically, an 80:10:10 ratio is used, but you can adjust it based on your requirements.

3. **Model Configuration**: Configure the mBERT model for text classification. This involves selecting the appropriate model architecture, setting hyperparameters, and fine-tuning if necessary.

4. **Tokenization and Encoding**: Tokenize the tweet text and encode it using the mBERT tokenizer. Ensure the input is in the required format for the model.

5. **Model Training**: Train the mBERT model on the training dataset. Monitor the performance on the validation set to avoid overfitting. Save the best model weights for later use.

6. **Model Evaluation**: Evaluate the trained model on the testing dataset to assess its performance. Calculate metrics such as accuracy, precision, recall, and F1-score to gauge the model's effectiveness.

## Evaluation

The model's performance can be evaluated using various metrics, including accuracy, precision, recall, and F1-score. These metrics provide insights into how well the model classifies the tweets into spam and ham categories.

## TODO 
1. make the streamlit Demo more cool and add the gif into hub after some sleeping :) 
## Streamlit Demo

A Streamlit demo is provided to interact with the trained model. To run the demo, execute the following command:

```
streamlit run demo.py
```

The Streamlit interface will open in your web browser. You can input your own tweets into the provided text box and click the "Classify" button to obtain predictions. The demo will display whether each tweet is classified as spam or ham.

## Contributing

Contributions to this project are welcome. If you would like to contribute, please follow the standard procedures for submitting pull requests. Ensure that any changes or additions align with the project's objectives and maintain the overall code quality.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to modify and distribute it as per the terms of the license.