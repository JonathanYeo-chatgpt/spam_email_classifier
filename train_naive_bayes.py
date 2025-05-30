import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time 

# Load your data
df_train = pd.read_csv("train_set.csv")  # Replace with your actual CSV file
df_test = pd.read_csv("test_set.csv")    # Replace with your actual CSV file

# Ensure no missing values in text column
df_train = df_train.dropna(subset=["cleaned_body", "spam"])
df_test = df_test.dropna(subset=["cleaned_body", "spam"])

# Features and labels
X_train = df_train["cleaned_body"].astype(str)  # Ensure text is string
Y_train = df_train["spam"].astype(int)  # Ensure labels are integers
X_test = df_test["cleaned_body"].astype(str)    # Ensure text is string
Y_test = df_test["spam"].astype(int)      # Ensure labels are integers



# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

start_time = time.time()
# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vectors, Y_train)

end_time = time.time()
print(f"Training completed in {end_time - start_time:.4f} seconds")
# Predict
y_pred = model.predict(X_test_vectors)

# Evaluation
print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Classification Report:\n", classification_report(Y_test, y_pred))



feature_names = vectorizer.get_feature_names_out()
spam_class_index = 1  # class label for spam

# Get the log probabilities of features for the spam class
spam_feature_log_probs = model.feature_log_prob_[spam_class_index]

# Sort and get top 20 indices
top_spam_indices = np.argsort(spam_feature_log_probs)[-20:][::-1]  # descending order

# Get corresponding words and their log probabilities
top_spam_words = pd.DataFrame({
    "word": feature_names[top_spam_indices],
    "log_probability": spam_feature_log_probs[top_spam_indices]
})

print(top_spam_words)


cm = confusion_matrix(Y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

joblib.dump(model, 'naive_bayes_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer_naive.pkl')