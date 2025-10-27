# Project Context

## Purpose
This project is an SMS spam detector. It uses a Naive Bayes machine learning model to classify messages as either "spam" (unsolicited or malicious) or "ham" (legitimate). The project includes a web application built with Streamlit that allows users to:
1.  Test the model with custom text input.
2.  View a random sample from the dataset and its prediction.
3.  See the model's performance metrics and visualizations.

The primary goal is to provide a simple and interactive tool to demonstrate the effectiveness of the Naive Bayes algorithm for text classification tasks.

## Tech Stack
- **Language:** Python
- **Web Framework:** Streamlit
- **Machine Learning:** Scikit-learn
- **Data Manipulation:** Pandas
- **Model Persistence:** Joblib
- **Data Visualization:** Matplotlib, Seaborn

## Project Conventions

### Code Style
- Code should be well-documented with comments explaining key logic.
- Use clear and descriptive variable and function names.
- The project uses a mix of English and Traditional Chinese for comments and UI text.

### Architecture Patterns
- **Frontend:** A single-page web application powered by Streamlit (`app.py`).
- **Model Training:** A separate script (`train_model.py`) is responsible for downloading the dataset, preprocessing the data, training the Naive Bayes model, and saving the model and vectorizer as `.pkl` files.
- **Model Loading:** The Streamlit app loads the pre-trained model and vectorizer for inference.
- **Visualization:** Helper scripts like `token_list.py` and `train_model.py` generate plots which are then displayed in the Streamlit app.

### Testing Strategy
- The model's performance is evaluated during the training phase (`train_model.py`).
- The dataset is split into an 80% training set and a 20% testing set.
- The model is evaluated against the test set using metrics like accuracy, precision, recall, and F1-score, summarized in a classification report.

### Git Workflow
[Describe your branching strategy and commit conventions]

## Domain Context
- **Domain:** Natural Language Processing (NLP), Spam Detection.
- **Key Concepts:**
    - **Spam:** Unwanted, often unsolicited, commercial or malicious messages.
    - **Ham:** Legitimate, desired messages.
    - **TF-IDF (Term Frequency-Inverse Document Frequency):** A numerical statistic used to reflect how important a word is to a document in a collection or corpus. This is used to convert text messages into numerical vectors that the model can understand.
    - **Naive Bayes:** A probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions between the features. It is a common and effective algorithm for text classification.

## Important Constraints
- **Dataset:** The model is trained exclusively on the "SMS Spam Collection Data Set" from UCI Machine Learning Repository, which contains English SMS messages.
- **Scope:** The model's effectiveness is likely limited to English SMS messages. It may not perform well on other types of text (e.g., emails, social media posts) or messages in other languages without retraining.

## External Dependencies
- **Dataset:** The training script downloads the dataset from a public GitHub repository URL. An internet connection is required to run `train_model.py` for the first time.