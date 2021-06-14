import argparse
import os
from collections import Counter

import logging
from typing import List

from perceptron_pos_tagger import Perceptron_POS_Tagger
from data_structures import Sentence, Corpus

SEED = 1234


def read_in_gold_data(filename):
    print(f"Reading gold data from {filename}")
    with open(filename) as f:
        lines = f.readlines()
        lines = [[tup.split('_') for tup in line.split()] for line in lines]
        sents = [Sentence(line) for line in lines]

    return sents


def read_in_plain_data(filename) -> List[Sentence]:
    print(f"Reading plain data from {filename}")
    with open(filename) as f:
        lines: List[str] = f.readlines()
        lines: List[List[str]] = [line.split() for line in lines]
        sents: List[Sentence] = [Sentence(line) for line in lines]

    return sents


def output_auto_data(auto_data, outfile):
    print(f"Writing results to {outfile}")
    with open(outfile, 'w') as f:
        f.write("\n".join([" ".join(line) for line in auto_data]))


def decode(model_dir: str, corpus: Corpus, outdir: str):
    if not (os.path.exists("auto")):
        os.mkdir("auto")
    if not (os.path.exists(outdir)):
        os.mkdir(outdir)
    sp_dir = os.path.join(outdir, "sp")
    ap_dir = os.path.join(outdir, "ap")
    if not (os.path.exists(sp_dir)):
        os.mkdir(sp_dir)
    if not (os.path.exists(ap_dir)):
        os.mkdir(ap_dir)
    model = Perceptron_POS_Tagger.from_disk(model_dir)
    auto_dev = model.tag(corpus.dev)
    auto_test = model.tag(corpus.test)

    output_auto_data(auto_dev, os.path.join(sp_dir, "dev.tagged"))
    output_auto_data(auto_test, os.path.join(sp_dir, "test.tagged"))

    auto_dev = model.avg_tag(corpus.dev, model.epoch)
    auto_test = model.avg_tag(corpus.test, model.epoch)

    output_auto_data(auto_dev, os.path.join(ap_dir, "dev.tagged"))
    output_auto_data(auto_test, os.path.join(ap_dir, "test.tagged"))



parser = argparse.ArgumentParser()
parser.add_argument("-a", "--all", help="Run all", action="store_true")
parser.add_argument("--average", help="Average Perceptron", action="store_true")
parser.add_argument("--debug", help="Debug Mode", action="store_true")
parser.add_argument("-w", "--window", help="Window Size", type=int, default=5)
parser.add_argument("-e", "--epochs", help="Number of epochs", type=int, default=5)
parser.add_argument("-v", "--vocab", help="Percentage of vocabulary", type=float, default=1.)
parser.add_argument("-i", "--input", help="Input Model", type=str, default="models/baseline")
parser.add_argument("-o", "--output", help="Output directory", type=str, default="auto/baseline/model.pkl")


if __name__ == '__main__':
    args = parser.parse_args()
    print(vars(args))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create console handler and set level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter(
        '%(asctime)s-%(name)s-%(levelname)s: %(message)s'
    )

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    train_file = os.path.join("train", "ptb_02-21.tagged")
    gold_dev_file = os.path.join("dev", "ptb_22.tagged")
    plain_dev_file = os.path.join("dev", "ptb_22.snt")
    test_file = os.path.join("test", "ptb_23.snt")

    # Read in data
    train_data = read_in_gold_data(train_file)
    gold_dev_data = read_in_gold_data(gold_dev_file)
    plain_dev_data = read_in_plain_data(plain_dev_file)
    test_data = read_in_plain_data(test_file)

    labels = []
    vocab = Counter()
    label_dict = Counter()
    for s in train_data:
        vocab.update(s.words)
        label_dict.update(s.tags)
    logger.info(f"Number of vocabs: {len(vocab)}")
    print(vocab.most_common(1000))
    print(f"Tags:\n")
    for k, v in label_dict.most_common(len(label_dict)):
        print(f"\ttag: {k}, count: {v}")
        labels.append(k)

    corpus = Corpus(
        train=train_data, dev=gold_dev_data, test=test_data,
        labels=labels
    )
    corpus.featurize()
    decode(args.input, corpus, args.output)
