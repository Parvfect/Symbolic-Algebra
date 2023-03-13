
# Encoding using byte pair - https://towardsdatascience.com/byte-pair-encoding-the-dark-horse-of-modern-nlp-eb36c7df4f10

import re
from collections import Counter, defaultdict

def build_vocab(corpus: str) -> dict:
    """ Building vocabulary from corpus """

    # Seperate each character in word by space and add mark end of token and then count frequency of each token

    return Counter([" ".join(word) + " </w>" for word in corpus.split()])

def get_stats(vocab: dict) -> dict:
    """ Get counts of  pairs of consecutive symbols """
    pairs = defaultdict(int) #defaultdict can handle missing keys
    for word, freq in vocab.items():
        symbols = word.split()

        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq

    return pairs

def merge_vocab(pair: tuple, v_in: dict) -> dict:
    """ Merge all occurances of the most frequent pair """

    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')#regex to find the bigram

    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    
    return v_out

vocab = build_vocab("What is my name huh?")
print(vocab)


