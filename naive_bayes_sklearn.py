import numpy as np
import pandas as pd

from sklearn import model_selection, naive_bayes, feature_extraction

def main():
    label_column = 'label'
    raw_data = pd.read_csv('./data/spam.csv', encoding='latin-1')

    processed_data = raw_data[['v1', 'v2']]
    processed_data = processed_data.rename(columns={'v1': label_column, 'v2': 'text'})
    processed_data[label_column] = processed_data.label.map({'ham': 0, 'spam': 1})
    processed_data['text'] = processed_data['text'].apply(process_text)

    train_raw, test_raw, train_labels, test_labels = model_selection.train_test_split(processed_data['text'], processed_data[label_column], test_size=0.1, random_state=10)
    
    cv = feature_extraction.text.CountVectorizer()
    train_points = cv.fit_transform(train_raw)
    test_points = cv.transform(test_raw)

    model = naive_bayes.MultinomialNB()
    model.fit(train_points, train_labels)

    accuracy = model.score(test_points, test_labels)
    print(f'Accuracy: {accuracy}')

def process_text(text, remove_chars=['.', ',', '!', '?', '\'', '(', ')', '"', ':', '-', '/', '\\', '$', '=', '>', '<', '&', '#', ';', '÷', '£', '+', '*']):
    text = text.lower()
    text = ''.join(c for c in text if c not in remove_chars and not c.isdigit())
    text = ' '.join(word.strip() for word in text.split())

    return text

def process_string(s, remove_chars=[]):
   return replace_chars(s.lower(), remove_chars).strip()

def replace_chars(s, remove_chars, replace_char=' '):
    return ''.join(replace_char if c in remove_chars else c for c in s.lower().strip())

if __name__ == "__main__":
    main()