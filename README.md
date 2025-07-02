Spam Classifier Project
Overview
This project implements a spam email classifier using Python and machine learning techniques. The notebook Spam_Classifier_model.ipynb processes a dataset of SMS messages, performs exploratory data analysis (EDA), and builds a spam classification model using Naive Bayes algorithms with TF-IDF vectorization.
Dataset
The dataset (spam.csv) contains SMS messages labeled as ham (non-spam) or spam. It includes the following columns:

v1: Label (ham or spam)
v2: Text of the message
Unnamed: 2, Unnamed: 3, Unnamed: 4: Unused columns (dropped during preprocessing)

Dependencies
The project requires the following Python libraries:

pandas
numpy
matplotlib
seaborn
scikit-learn

Install dependencies using:
pip install pandas numpy matplotlib seaborn scikit-learn

Notebook Structure

Data Loading and Preprocessing:

Loads the dataset using pandas.
Drops irrelevant columns (Unnamed: 2, Unnamed: 3, Unnamed: 4).
Renames columns: v1 to message_type, v2 to message.
Encodes message_type using LabelEncoder (ham = 0, spam = 1).
Removes 403 duplicate entries.
Resets the index for consistency.


Exploratory Data Analysis (EDA):

Displays dataset information and checks for missing values (none found).
Visualizes the top 30 spam words using a bar plot (requires common_words_df, which is not defined in the provided code snippet).


Model Building:

Uses TfidfVectorizer to convert text messages into numerical features (max_features=3000).
Splits data into training (80%) and testing (20%) sets.
Evaluates three Naive Bayes models:
GaussianNB: Accuracy ~86.36%, Precision ~0.45
MultinomialNB: Accuracy ~97.00%, Precision ~1.0
BernoulliNB: Accuracy ~97.78%, Precision ~0.99


Selects MultinomialNB for its highest precision score (no false positives).



Key Findings

Multinomial Naive Bayes with TF-IDF vectorization yields the best performance, achieving a precision score of 1.0, ensuring no false positives.
The dataset is cleaned by removing duplicates and unused columns, ensuring robust preprocessing.

Usage

Clone the repository:git clone <repository_url>


Place the spam.csv dataset in the project directory.
Open the Spam_Classifier_model.ipynb notebook in Jupyter.
Run the cells sequentially to preprocess data, perform EDA, and train the model.

Notes

The common_words_df used in the EDA visualization is not defined in the provided notebook. Ensure it is created (e.g., by analyzing word frequencies in spam messages) for the bar plot to work.
The file path for spam.csv is hardcoded (C:\Users\ohkba\Downloads\spam (1).csv). Update it to match your local environment.
The notebook assumes a Python 3.9 environment.

Future Improvements

Add word frequency analysis to generate common_words_df for EDA.
Explore additional text preprocessing (e.g., stopword removal, stemming).
Test other machine learning models (e.g., SVM, Random Forest) for comparison.
Implement cross-validation for more robust model evaluation.

License
This project is licensed under the MIT License.
