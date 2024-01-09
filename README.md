Data Preparation:
Reads a CSV file containing spam and non-spam (ham) messages.
Cleans the data, removing unnecessary columns and renaming columns for clarity.
Encodes the 'Target' column (ham/spam) into numerical values.

Feature Engineering:
Calculates various text-based features like number of characters, words, and sentences.
Visualizes these features using histograms and statistical descriptions.

Text Preprocessing:
Converts text to lowercase and tokenizes it (breaks it into words).
Removes punctuation, stopwords, and non-alphabetic characters.
Stems words (reduces them to their root form).

Visualization:
Generates word clouds for spam and non-spam messages to display the most frequent words in each category.

Statistical Analysis:
Identifies the most common words in spam and non-spam messages using frequency counts.

Machine Learning Model Building:
Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text into numerical features.
Splits the data into training and testing sets.
Builds a Multinomial Naive Bayes classifier.
Evaluates the model's performance using accuracy, precision, and confusion matrix metrics.

Model Serialization:
Saves the trained model and vectorizer for future use.

app.py:
This script sets up a Streamlit app for email/SMS spam detection. It imports necessary libraries, including Streamlit for the web app interface, loads a Lottie animation for visual appeal, and initializes NLTK tools for text processing. Using a pre-trained model and vectorizer loaded from pickle files, it defines a function to transform user-input text by cleaning, tokenizing, removing stopwords and punctuation, and stemming words. The Streamlit app presents a text area for user input, allowing them to enter a message for spam detection. Upon clicking the 'Predict' button, it processes the input, applies the trained model, and displays whether the message is identified as spam or not. Error handling is implemented for various stages, ensuring smooth functionality of the app even in case of exceptions.
