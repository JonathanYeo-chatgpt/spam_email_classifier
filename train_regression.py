import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time

# Load training and test data
train_df = pd.read_csv('train_set.csv')
test_df = pd.read_csv('test_set.csv')


train_df = train_df.dropna(subset=['cleaned_body', 'spam'])
test_df = test_df.dropna(subset=['cleaned_body', 'spam'])

# Extract features and labels
X_train_texts = train_df['cleaned_body'].astype(str)
y_train = train_df['spam']
X_test_texts = test_df['cleaned_body'].astype(str)
y_test = test_df['spam']

# Convert text to TF-IDF features

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vectors = vectorizer.fit_transform(X_train_texts)
X_test_vectors = vectorizer.transform(X_test_texts)

start_time = time.time()  

# Train classifier
clf = LogisticRegression()
clf.fit(X_train_vectors, y_train)

end_time = time.time()
print(f"Training completed in {end_time - start_time:.4f} seconds")

# Predict on test set
y_pred = clf.predict(X_test_vectors)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Get feature names from vectorizer
feature_names = vectorizer.get_feature_names_out()

# Get coefficients from the trained logistic regression model
# coef_ is shape (1, n_features) for binary classification
coef = clf.coef_[0]

# Sort indices by coefficient value descending to get words strongly associated with spam
top_spam_indices = np.argsort(coef)[-20:][::-1]

# Create a dataframe with word and coefficient
top_spam_words = pd.DataFrame({
    "word": feature_names[top_spam_indices],
    "coefficient": coef[top_spam_indices]
})




print(top_spam_words)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

joblib.dump(clf, 'logistic_regression_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer_regression.pkl')
