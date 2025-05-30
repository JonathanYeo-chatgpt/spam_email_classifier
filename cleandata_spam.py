import os
import re
import pandas as pd
from email import policy
from email.parser import BytesParser
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def is_probably_html(text):
    return bool(re.search(r'<[^>]+>', text))

def extract_headers_and_body(filepath):
    with open(filepath, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)

    # Try to get the preferred body (plain, fallback html)
    body_part = msg.get_body(preferencelist=('plain', 'html'))
    if not body_part:
        return {
            'From': msg.get('From'),
            'Subject': msg.get('Subject'),
        }, ""

    # Decode body payload safely with fallback charset
    payload = body_part.get_payload(decode=True)
    charset = body_part.get_content_charset()
    try:
        if charset:
            body_text = payload.decode(charset)
        else:
            body_text = payload.decode('utf-8')
    except (LookupError, UnicodeDecodeError):
        body_text = payload.decode('latin1', errors='replace')

    # If HTML, strip tags
    if body_part.get_content_type() == 'text/html':
        soup = BeautifulSoup(body_text, 'html.parser')
        body_text = soup.get_text(separator=' ', strip=True)

    return {
        'From': msg.get('From'),
        'Subject': msg.get('Subject'),
    }, body_text

def clean_email_body(text):
    if is_probably_html(text):
        soup = BeautifulSoup(text, 'html.parser')
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        text = soup.get_text(separator=' ')
    # Parse HTML and remove all tags and inline CSS
    
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

def process_eml_files(directory):
    cleaned_data = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            headers, body = extract_headers_and_body(filepath)
            cleaned_body = clean_email_body(body)
            

            cleaned_data.append({
                'filename': filename,
                'from': headers['From'],
                'subject': headers['Subject'],
                'cleaned_body': cleaned_body,
                'spam': 1  
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return cleaned_data

def main():
    folder_path = 'archive/spam' 
    cleaned_emails = process_eml_files(folder_path)
    df = pd.DataFrame(cleaned_emails)
    df.to_csv('cleaned_email_spam.csv', index=False)
    
if __name__ == "__main__":
    main()
