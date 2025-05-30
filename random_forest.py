import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
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

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

start_time = time.time()



# Train the model
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth = None, max_features = 'log2', min_samples_leaf = 1, min_samples_split = 2)
rf_model.fit(X_train_vectors, Y_train)


end_time = time.time()
print(f"Training completed in {end_time - start_time:.4f} seconds")

# Predict
rf_preds = rf_model.predict(X_test_vectors)

# Evaluate

print("Accuracy:", accuracy_score(Y_test, rf_preds))
print("Classification Report:\n", classification_report(Y_test, rf_preds))


importances = rf_model.feature_importances_
feature_names = vectorizer.get_feature_names_out()

# Combine words with their importance scores
feature_importance_df = pd.DataFrame({
    "word": feature_names,
    "importance": importances
})

# Sort and show top 20 most important words
top_words = feature_importance_df.sort_values(by="importance", ascending=False).head(20)
print("\nTop 20 Important Words for Spam Detection:\n")
print(top_words)

cm = confusion_matrix(Y_test, rf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot(cmap='Blues')
plt.show()

joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer_forest.pkl')

