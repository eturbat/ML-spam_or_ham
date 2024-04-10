import re
import math
import string
import pandas as pd
import nltk
from collections import defaultdict
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing function for the simple model
def preprocess_simple(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return words

# Preprocessing function for the improved model
def preprocess_improved(text):
    stemmer = SnowballStemmer("english")
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()

    words = [stemmer.stem(word) for word in words if word not in stop_words]

    return ' '.join(words)

# Training function for the simple Naive Bayes model
def train_naive_bayes(emails, labels):
    num_emails = len(emails)
    num_spams = sum(labels)
    num_hams = num_emails - num_spams

    word_counts_spam = defaultdict(int)
    word_counts_ham = defaultdict(int)

    for i in range(num_emails):
        words = preprocess_simple(emails[i])
        for word in words:
            if labels[i] == 1:
                word_counts_spam[word] += 1
            else:
                word_counts_ham[word] += 1

    word_probs_spam = {word: (word_counts_spam[word] + 1) / (num_spams + 1) for word in word_counts_spam}
    word_probs_ham = {word: (word_counts_ham[word] + 1) / (num_hams + 1) for word in word_counts_ham}

    return word_probs_spam, word_probs_ham

# Classification function for the simple Naive Bayes model
def classify(email, word_probs_spam, word_probs_ham):
    words = preprocess_simple(email)

    spam_score = 0
    ham_score = 0

    for word in words:
        if word in word_probs_spam:
            spam_score += math.log(word_probs_spam[word])
        if word in word_probs_ham:
            ham_score += math.log(word_probs_ham[word])

    return 1 if spam_score > ham_score else 0

# Load dataset
data = pd.read_csv('spam.csv')

# Extract email texts and labels
emails = data['v2'].tolist()
labels = [1 if label == 'spam' else 0 for label in data['v1'].tolist()]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

# Train the simple Naive Bayes model
word_probs_spam, word_probs_ham = train_naive_bayes(X_train, y_train)

# Test the simple Naive Bayes model
correct = 0
total = len(X_test)
for i in range(total):
    prediction = classify(X_test[i], word_probs_spam, word_probs_ham)
    if prediction == y_test[i]:
        correct += 1

# Calculate and print the accuracy for the simple model
accuracy = correct / total
print("Simple model accuracy: {:.2f}%".format(accuracy * 100))

# Classify the test data using the simple model and get the predicted labels
y_pred_simple = []
for email in X_test:
    y_pred_simple.append(classify(email, word_probs_spam, word_probs_ham))

# Generate the confusion matrix for the simple model
cm_simple = confusion_matrix(y_test, y_pred_simple)

# Download the stopwords
nltk.download('stopwords')

# Preprocess the emails using the improved preprocessing function
emails_improved = [preprocess_improved(email) for email in emails]

# Vectorize the emails using the TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails_improved)

# Split the dataset into training and testing sets for the improved model
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train the improved model using the MultinomialNB classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Test the improved model
y_pred_improved = classifier.predict(X_test)

# Calculate and print the accuracy for the improved model
accuracy_improved = accuracy_score(y_test, y_pred_improved)
print("Improved model accuracy: {:.2f}%".format(accuracy_improved * 100))

# Generate the confusion matrix for the improved model
cm_improved = confusion_matrix(y_test, y_pred_improved)

# Plot the confusion matrices as heatmaps for both models
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_simple, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_xlabel('Predicted label')
axes[0].set_ylabel('True label')
axes[0].set_title('Simple Model Confusion Matrix')

sns.heatmap(cm_improved, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_xlabel('Predicted label')
axes[1].set_ylabel('True label')
axes[1].set_title('Improved Model Confusion Matrix')

plt.show()

