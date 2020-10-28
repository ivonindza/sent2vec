#!/usr/bin/env python3
""" Benchmark sent2vec with 1M sentences.
"""
import argparse
import os.path
from timeit import default_timer as timer

import sent2vec


MODEL_DIR = '/models'
SENTENCES_FILE = '/sentences/sentences'

def modelpath(model_name):
    path = os.path.join(MODEL_DIR, model_name + '.bin')
    if not os.path.isfile(path):
        msg = f'Model {model_name} could not be found'
        raise argparse.ArgumentTypeError(msg)
    return path

def slices(l: list, n: int):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=modelpath, help="sent2vec model")
    args = parser.parse_args()

    with open(SENTENCES_FILE, 'r', errors='ignore') as f:
        sentences = f.read().splitlines()

    model = sent2vec.Sent2vecModel()
    model.load_model(args.model, mmaped_io=True)
    start_time = timer()
    for s in slices(sentences, 1000):
        embs = model.embed_sentences(s)
    duration = timer() - start_time
    print(f'Runtime: {duration}')

if __name__ == '__main__':
    main()
