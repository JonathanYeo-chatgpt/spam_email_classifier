import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

# Load training and test data
train_df = pd.read_csv('train_set.csv')
test_df = pd.read_csv('test_set.csv')

# Drop rows with missing data
train_df = train_df.dropna(subset=['cleaned_body', 'spam'])
test_df = test_df.dropna(subset=['cleaned_body', 'spam'])

# Extract features and labels
X_train_texts = train_df['cleaned_body'].astype(str)
y_train = train_df['spam'].astype(int)
X_test_texts = test_df['cleaned_body'].astype(str)
y_test = test_df['spam'].astype(int)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vectors = vectorizer.fit_transform(X_train_texts)
X_test_vectors = vectorizer.transform(X_test_texts)

# Train 5 times
training_times = []
accuracies = []
best_model = None
best_preds = None
best_accuracy = 0

for i in range(5):
    print(f"\nTraining Run {i+1}...")
    start_time = time.time()

    clf = LogisticRegression()
    clf.fit(X_train_vectors, y_train)

    end_time = time.time()
    duration = end_time - start_time
    training_times.append(duration)

    y_pred = clf.predict(X_test_vectors)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"Training Time: {duration:.4f} seconds")
    print(f"Accuracy: {acc:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = clf
        best_preds = y_pred

# Report average results
print("\n--- Summary ---")
print(f"Average Training Time: {np.mean(training_times):.4f} seconds")
print(f"Average Accuracy: {np.mean(accuracies):.4f}")

# Final evaluation
print("\nClassification Report of Best Model:\n", classification_report(y_test, best_preds))

# Top 20 spam-indicative words based on highest positive coefficients
feature_names = vectorizer.get_feature_names_out()
coef = best_model.coef_[0]
top_spam_indices = np.argsort(coef)[-20:][::-1]

top_spam_words = pd.DataFrame({
    "word": feature_names[top_spam_indices],
    "coefficient": coef[top_spam_indices]
})

print("\nTop 20 Spam-Indicative Words:\n")
print(top_spam_words)

# Confusion matrix
cm = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Save best model and vectorizer
joblib.dump(best_model, 'logistic_regression_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer_regression.pkl')
