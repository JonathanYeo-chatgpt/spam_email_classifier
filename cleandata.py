import pandas as pd
import os
import email
import re
from email import policy
from nltk.corpus import stopwords
from email.parser import BytesParser
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def is_probably_html(text):
    return bool(re.search(r'<[^>]+>', text))


def extract_email_parts(raw_email):
    headers, body = raw_email.split('\n\n', 1)  # Split header and body
    return headers.strip(), body.strip()


def get_header_fields(header_text):
    msg = email.message_from_string(header_text)
    return {
        'From': msg.get('From'),
        'To': msg.get('To'),
        'Subject': msg.get('Subject'),
        'Date': msg.get('Date')
    }


def clean_email_body(text):
    # Parse HTML and remove all tags and inline CSS
    if is_probably_html(text):
        soup = BeautifulSoup(text, 'html.parser')
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        text = soup.get_text(separator=' ')

    # Get only visible text


    # Remove mailing list footers
    text = re.split(r"_{5,}|^--\s*$|^Sent from", text, flags=re.MULTILINE)[0]

    # Remove email reply headers
    text = re.sub(r'^.*?From:.*?\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.*?Date:.*?\n', '', text, flags=re.MULTILINE)

    # Remove quoted lines
    text = re.sub(r'(?m)^>.*$', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove non-text (punctuation, numbers)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Lowercase
    text = text.lower()

    # Tokenize, remove stopwords, and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

    return ' '.join(words)

def actual_cleaning(file_dir):
    cleaned_data = []
    for filename in os.listdir(file_dir):
      with open(os.path.join(file_dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
          raw = f.read()
          headers, body = extract_email_parts(raw)
          fields = get_header_fields(headers)
          cleaned_body = clean_email_body(body)
          cleaned_data.append({
              'filename': filename,
              'from': fields['From'],
              'subject': fields['Subject'],
              'cleaned_body': cleaned_body,
              'spam': 0
          })
    return cleaned_data

def main():
  cleaned_data = []

  cleaned_data.extend(actual_cleaning("archive/easy_ham/easy_ham"))


  

  df = pd.DataFrame(cleaned_data)
  df.to_csv('cleaned_emails.csv', index=False)


if __name__ == "__main__":
    # Example usage
    main()