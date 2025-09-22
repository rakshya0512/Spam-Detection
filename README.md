# Spam-Detection
A deep learning-based spam detection system for emails and SMS messages using Bi-LSTM with Attention. Classifies messages as spam or ham with high accuracy.
# Description:
This project implements a deep learning-based spam detection system for emails and SMS messages. Using a Bi-directional LSTM (Bi-LSTM) with Attention, the model classifies messages as spam or ham (non-spam). The system captures sequential patterns in text and highlights important words to improve detection accuracy.

# Key Features:

Preprocessing and tokenization of text data.

Bi-LSTM with Attention mechanism for context-aware spam detection.

High-performance model achieving over 98% accuracy on the test dataset.

Visualization of training/validation loss and confusion matrix.

Model and tokenizer saved for deployment and prediction on new messages.

# Technologies Used:

Python, TensorFlow/Keras, NumPy, Pandas, scikit-learn, Matplotlib, Joblib

# Usage:

Load dataset (spam.csv) with text and label columns.

Preprocess and tokenize messages.

Train the Bi-LSTM with Attention model.

Evaluate performance and save the model/tokenizer.

Use the saved model to predict spam in new messages.

# Applications:

Real-time email and SMS spam filtering.

Preprocessing for NLP pipelines and text classification tasks.

# Dataset Reference:

SMS Spam Collection Dataset ( https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download )
