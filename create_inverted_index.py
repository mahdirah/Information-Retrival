import numpy as np
import collections

K = 0.5

def generate_ii(dict):
    vocabs =  {word: [] for word in np.unique(np.concatenate(list(dict.values())))}
    dicts_keys = dict.keys()
    for key in dicts_keys:
        u_vocab, counts = np.unique(dict[key], return_counts = True)
        for i in range(len(u_vocab)):
            vocabs[u_vocab[i]].append({'docID': key, 'frequency': counts[i]})
    od = collections.OrderedDict(sorted(vocabs.items()))
    return od

def generate_tf_idf_champions(dicts, docs):
    N = len(docs)
    tf_idf = np.zeros([len(dicts.keys()), N])
    terms = list(dicts.keys())
    champions_list = np.zeros([len(dicts.keys()), N])
    for i in range(tf_idf.shape[0]):
        term = terms[i]
        f_t = dicts[term]
        idf = np.log10(N / len(f_t))
        for doc in f_t:
            docID = docs.index(doc['docID'])
            tf = 1 + np.log10(doc['frequency'])
            if (tf * idf > K):
                champions_list[i][docID] = tf * idf
            tf_idf[i][docID] = tf * idf
    return tf_idf, champions_list

