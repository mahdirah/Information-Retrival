import datetime
from os import listdir
from os.path import isfile, join
from stemming import *
from create_inverted_index import *
from search import *
import numpy as np


data_path = "docs"
documents_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

def extract_dictionaries(file_names):
    files = {}
    raw_docs = {}
    for file in file_names:
        f = open(data_path + '/' + file)
        raw_data = f.read()
        raw_docs.update({file: raw_data.replace('\n', '')})
        files.update({file: np.array(raw_data.
                     replace('\n', ' ').
                     replace('\u200c', ' ').
                     replace('.', '').split(' '))})
        f.close()
    return files, raw_docs

a = datetime.datetime.now()
dicts, raw_documents = extract_dictionaries(documents_files)
dicts = stemming_5(stemming_4(stemming_3(stemming_2(stemming_1(dicts)))))
dicts = {key: [v for v in val if v != ''] for key, val in dicts.items()}
inverted_index = generate_ii(dicts)
tf_idf, champions = generate_tf_idf_champions(inverted_index, documents_files)
b = datetime.datetime.now()
print('Time spent: ', (b - a).seconds, " seconds and ", (b - a).microseconds, " microseconds")
x = input("Enter query: ")
c = datetime.datetime.now()
search_by_tfidf(tf_idf, documents_files, x, inverted_index)
d = datetime.datetime.now()
print(f'Time spent: {(d - c).seconds} seconds and {(d - c).microseconds} microseconds')
