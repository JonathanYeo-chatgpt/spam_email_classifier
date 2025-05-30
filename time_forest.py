import pandas as pd
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
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

# Train 5 times and collect metrics
training_times = []
accuracies = []
best_model = None
best_accuracy = 0

for i in range(5):
    print(f"\nTraining Run {i+1}...")
    start_time = time.time()

    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=None,
        max_features='log2',
        min_samples_leaf=1,
        min_samples_split=2
    )
    rf_model.fit(X_train_vectors, Y_train)

    end_time = time.time()
    duration = end_time - start_time
    training_times.append(duration)

    # Predict and evaluate
    rf_preds = rf_model.predict(X_test_vectors)
    acc = accuracy_score(Y_test, rf_preds)
    accuracies.append(acc)

    print(f"Training Time: {duration:.2f} seconds")
    print(f"Accuracy: {acc:.4f}")

    # Keep the best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = rf_model
        best_preds = rf_preds

# Report averages
print("\n--- Summary ---")
print(f"Average Training Time: {np.mean(training_times):.2f} seconds")
print(f"Average Accuracy: {np.mean(accuracies):.4f}")

# Final evaluation of the best model
print("\nClassification Report of Best Model:\n", classification_report(Y_test, best_preds))

# Feature importances from best model
importances = best_model.feature_importances_
feature_names = vectorizer.get_feature_names_out()
feature_importance_df = pd.DataFrame({
    "word": feature_names,
    "importance": importances
})

top_words = feature_importance_df.sort_values(by="importance", ascending=False).head(20)
print("\nTop 20 Important Words for Spam Detection:\n")
print(top_words)

# Confusion matrix for best model
cm = confusion_matrix(Y_test, best_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot(cmap='Blues')
plt.show()

# Save best model and vectorizer
joblib.dump(best_model, 'random_forest_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer_forest.pkl')
