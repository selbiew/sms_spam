import collections
import math
import functools
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    label_column = 'label'
    raw_data = pd.read_csv('./data/spam.csv', encoding='latin-1')

    processed_data = raw_data[['v1', 'v2']].iloc[:1500]
    processed_data = processed_data.rename(columns={'v1': label_column, 'v2': 'text'})
    processed_data[label_column] = processed_data.label.map({'ham': 0, 'spam': 1})
    processed_data['text'] = processed_data['text'].apply(process_text)

    vocabulary = get_vocabulary(processed_data['text'].str.cat(sep=' '))
    processed_data['feature_list'] = processed_data['text'].apply(lambda s: [vocabulary[word] for word in s.split()])

    ham, spam = processed_data[processed_data[label_column] == 0]['feature_list'], processed_data[processed_data[label_column] == 1]['feature_list']
    ham_probs, spam_probs = get_probabilities(ham, spam, vocabulary)

    processed_data['prediction'] = processed_data['feature_list'].apply(lambda fs: predict(fs, ham_probs, spam_probs, ham, spam, vocabulary))
    processed_data['correct'] = np.where(processed_data[label_column] == processed_data['prediction'], 1, 0)

    accuracy = processed_data['correct'].mean()
    print(f'Accuracy: {accuracy}')

def predict(fs, neg_probabilities, pos_probabilities, negatives, positives, vocabulary):
    neg_wordcount = sum(len(fs) for fs in negatives)
    neg_prob = functools.reduce(operator.mul, (neg_probabilities[f] if f in neg_probabilities else (1 / (neg_wordcount + len(vocabulary))) for f in fs))
    pos_wordcount = sum(len(fs) for fs in positives)
    pos_prob = functools.reduce(operator.mul, (pos_probabilities[f] if f in pos_probabilities else (1 / (pos_wordcount + len(vocabulary))) for f in fs))

    return 0 if neg_prob > pos_prob else 1

def get_probabilities(negatives, positives, vocabulary):
    neg_wordcount = sum(len(fs) for fs in negatives)
    neg_probabilities = {k: (v + 1) / (neg_wordcount + len(vocabulary)) for k, v in functools.reduce(operator.add, (collections.Counter(fs) for fs in negatives)).items()}
    
    pos_wordcount = sum(len(fs) for fs in positives)
    pos_probabilities = {k: (v + 1) / (pos_wordcount + len(vocabulary)) for k, v in functools.reduce(operator.add, (collections.Counter(fs) for fs in positives)).items()}

    return neg_probabilities, pos_probabilities

def get_vocabulary(s):
    return {word: i + 1 for i, word in enumerate(set(s.split()))}

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