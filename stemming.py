import numpy as np
import string
import pandas as pd
import re

## Remove 3 most frequent term
def stemming_1(dics):
    words = np.concatenate(list(dics.values()))
    for i in range(3):
        unique, pos = np.unique(words, return_inverse=True)
        counts = np.bincount(pos)
        maxpos = counts.argmax()
        words = np.delete(words, np.argwhere(words == unique[maxpos]))
        dics = {key: np.delete(val, np.argwhere(np.array(val) == unique[maxpos])) for key, val in dics.items()}
    return dics


## Remove punctuation
def stemming_2(dics):
    exclude = set(string.punctuation + '؟' + '«')
    dics = {key: [''.join(ch for ch in s if ch not in exclude) for s in val] for key, val in dics.items()}
    return dics


## Remove ها and تر and ترین
def stemming_3(dics):
    special_chars = np.array(['ترین', 'تر', 'ها', 'به'])
    for sc in special_chars:
        try:
            dics = {key: np.char.replace(val, sc, '') for key, val in dics.items()}
        except TypeError:
            pass
    return dics

# Convert دان
def stemming_4(dics):
    Danestan_regex = '^\u0645\u06CC\u0020\u062F\u0627\u0646*'
    for key, values in dics.items():
        for value in values:
            if re.search(Danestan_regex, value):
                if not 'دان' in dics[key]:
                    dics[key] = np.append(dics[key], 'دان')
                dics[key] = dics[key][dics[key] != value]
    return dics

## Conver persian numbers
def stemming_5(dics):
    persian_numbers = ['۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹', '۰',]
    english_numbers = np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'])
    dics = {key: ["".join([dict(zip(persian_numbers, english_numbers)).get(x, x) for x in s]) for s in val]
            for key, val in dics.items()}
    return dics