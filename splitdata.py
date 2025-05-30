import pandas as pd
from sklearn.model_selection import train_test_split
import csv


with open('cleaned_email_spam.csv', 'r', encoding="utf-8") as f1, open('cleaned_emails.csv', 'r', encoding="utf-8") as f2, open("filtered_output.csv", 'w', encoding="utf-8", newline='') as output:
    reader1 = csv.reader(f1)
    reader2 = csv.reader(f2)
    writer = csv.writer(output)

    # Write the header and content of file1
    for i, row in enumerate(reader1):
        writer.writerow(row)

    # Skip the header of file2
    next(reader2)
    
    # Write the rest of file2
    for row in reader2:
        writer.writerow(row)

   


    


df = pd.read_csv('filtered_output.csv')

# Split into features (X) and labels (y)
X = df.drop(columns=['spam'])  # everything except the 'spam' column
y = df['spam']                 # the target column


# Split into training and test sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify keeps spam/ham ratio consistent
)

# Optional: Save the splits to CSV
X_train.assign(spam=y_train).to_csv('train_set.csv', index=False)
X_test.assign(spam=y_test).to_csv('test_set.csv', index=False)

print("Train and test sets created and saved.")
