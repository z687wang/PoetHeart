import numpy as np

def get_data():
    train_data_files = ['Shelley.txt', 'blake.txt']

    text = ''

    for file in train_data_files:
        with open(file, 'r') as f:
            text = text + f.read()

    vocab = sorted(set(text))
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

    return text, encoded, vocab, int_to_vocab, vocab_to_int