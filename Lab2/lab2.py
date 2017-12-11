from typing import List, Set

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize


def tokenize_and_clean(text: str, stop: Set[str]) -> List[str]:
    return [w for w in wordpunct_tokenize(text.lower()) if w.lower() not in stop]


def get_dataset(filename: str):
    nltk.download('stopwords')
    stop = set(stopwords.words('english'))
    stop.update([c for c in '.,"\'?!:;()[]{}'] + ['\x92'])

    data = []
    with open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            if not line.strip():
                continue

            text, label = line.strip().rsplit(',', 1)
            if label not in {'spam', 'ham'}:
                raise ValueError('Unknown label: ' + label)

            text = tokenize_and_clean(text, stop)
            data.append((text, label,))
    return data


if __name__ == '__main__':
    data = get_dataset('corpus/english.txt')
    print(data)



