import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

# Load your data
df_train = pd.read_csv("train_set.csv")
df_test = pd.read_csv("test_set.csv")

# Ensure no missing values in text column
df_train = df_train.dropna(subset=["cleaned_body", "spam"])
df_test = df_test.dropna(subset=["cleaned_body", "spam"])

# Features and labels
X_train = df_train["cleaned_body"].astype(str)
Y_train = df_train["spam"].astype(int)
X_test = df_test["cleaned_body"].astype(str)
Y_test = df_test["spam"].astype(int)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Training loop
training_times = []
accuracies = []
best_model = None
best_accuracy = 0
best_preds = None

for i in range(5):
    print(f"\nTraining Run {i+1}...")
    start_time = time.time()

    model = MultinomialNB()
    model.fit(X_train_vectors, Y_train)

    end_time = time.time()
    duration = end_time - start_time
    training_times.append(duration)

    y_pred = model.predict(X_test_vectors)
    acc = accuracy_score(Y_test, y_pred)
    accuracies.append(acc)

    print(f"Training Time: {duration:.4f} seconds")
    print(f"Accuracy: {acc:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_preds = y_pred

# Report averages
print("\n--- Summary ---")
print(f"Average Training Time: {np.mean(training_times):.4f} seconds")
print(f"Average Accuracy: {np.mean(accuracies):.4f}")

# Final evaluation of the best model
print("\nClassification Report of Best Model:\n", classification_report(Y_test, best_preds))

# Top 20 spam-indicative words
feature_names = vectorizer.get_feature_names_out()
spam_class_index = 1  # spam class index
spam_feature_log_probs = best_model.feature_log_prob_[spam_class_index]
top_spam_indices = np.argsort(spam_feature_log_probs)[-20:][::-1]

top_spam_words = pd.DataFrame({
    "word": feature_names[top_spam_indices],
    "log_probability": spam_feature_log_probs[top_spam_indices]
})

print("\nTop 20 Spam-Indicative Words:\n")
print(top_spam_words)

# Confusion matrix
cm = confusion_matrix(Y_test, best_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Save best model and vectorizer
joblib.dump(best_model, 'naive_bayes_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer_naive.pkl')
