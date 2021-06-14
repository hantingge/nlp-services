import argparse
import copy
import os
from collections import Counter

import sys
import logging
import random
from typing import List

from perceptron_pos_tagger import Perceptron_POS_Tagger
from data_structures import Sentence, Corpus

__author__ = "Hanting Ge"
LOGGER = logging.getLogger(__name__)
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


def run_all(corpus: Corpus, debug: bool, epochs: int, outdir: str, model_dir: str):
    logger.info("Running Ablation study")
    if debug:
        outdir = "debug_files"
        model_dir = "debug_files"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    params = [
        None, "prev_tag", "curr", "prev", "next", "ling"
    ]
    #default = {k: True for k in params}
    logger.debug(f"Baseline parameters: {params}")
    for i in range(len(params)):
        remove_f = params[i]
        if remove_f in [None, "prev_tag"]:
            corpus.featurize()
        else:
            corpus.featurize(remove_feature=remove_f)
        logger.info(
            f"Sample features:\n{random.sample(corpus.train[0].features, 1)}"
        )
        exp_name = "baseline" if remove_f is None else "-".join(["baseline", remove_f])

        model_dir = os.path.join(model_dir, exp_name)
        outdir = os.path.join(outdir, exp_name)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        logger.info(f"Model Directory: {model_dir}")
        logger.info(f"Output Directory: {outdir}")

        my_tagger = Perceptron_POS_Tagger(corpus.labels, model_dir)
        use_prev_tag = False if remove_f == "prev_tag" else True
        if debug:
            print("Debug Mode")
            my_tagger.train(
                random.sample(corpus.train, 128*5), corpus.dev,
                epochs=epochs, batch_size=128, use_prev_tag=use_prev_tag
            )
        else:
            my_tagger.train(
                corpus.train, corpus.dev, epochs=epochs,
                batch_size=256, use_prev_tag=use_prev_tag
            )

        logger.info("Decoding using structured perceptron")
        # Apply your tagger on dev & test data
        auto_dev_data = my_tagger.tag(corpus.dev)
        auto_test_data = my_tagger.tag(corpus.test)

        sp_dir = os.path.join(outdir, "sp")
        ap_dir = os.path.join(outdir, "ap")
        if not (os.path.exists(sp_dir)):
            os.mkdir(sp_dir)
        if not (os.path.exists(ap_dir)):
            os.mkdir(ap_dir)

        devfile = os.path.join(sp_dir, "dev.tagged")
        testfile = os.path.join(sp_dir, "test.tagged")
        # Output your auto tagged data
        output_auto_data(auto_dev_data, devfile)
        output_auto_data(auto_test_data, testfile)

        logger.info("Decoding using Averaged Perceptron")
        # Apply your tagger on dev & test data
        auto_dev_data = my_tagger.avg_tag(corpus.dev, args.epochs)
        auto_test_data = my_tagger.avg_tag(corpus.test, args.epochs)

        devfile = os.path.join(ap_dir, "dev.tagged")
        testfile = os.path.join(ap_dir, "test.tagged")
        # Outpur your auto tagged data
        output_auto_data(auto_dev_data, devfile)
        output_auto_data(auto_test_data, testfile)


def run(corpus, remove_feature, epochs, outdir, model_dir):
    corpus.featurize(remove_feature=remove_feature)
    logger.info(f"Output Directory: {outdir}")
    logger.info(f"Model Directory: {model_dir}")
    experiment = "baseline" if remove_feature is None else "-".join(["baseline", remove_feature])
    outdir = os.path.join(outdir, experiment)
    model_dir = os.path.join(model_dir, experiment)
    logger.info(f"Output Directory: {outdir}")
    logger.info(f"Model Directory: {model_dir}")
    model = Perceptron_POS_Tagger(corpus.labels, output=model_dir)
    use_prev_tag = False if remove_feature=="prev_tag" else True
    model.train(corpus.train, corpus.dev, epochs=epochs,
                batch_size=256, use_prev_tag=use_prev_tag)

    sp_dir = os.path.join(outdir, "sp")
    ap_dir = os.path.join(outdir, "ap")
    if not (os.path.exists(sp_dir)):
        os.mkdir(sp_dir)
    if not (os.path.exists(ap_dir)):
        os.mkdir(ap_dir)
    logger.info("Decoding using structured perceptron")
    # Apply your tagger on dev & test data
    auto_dev_data = model.tag(corpus.dev)
    auto_test_data = model.tag(corpus.test)

    devfile = os.path.join(sp_dir, "dev.tagged")
    testfile = os.path.join(sp_dir, "test.tagged")
    # Output your auto tagged data
    output_auto_data(auto_dev_data, devfile)
    output_auto_data(auto_test_data, testfile)

    logger.info("Decoding using Averaged Perceptron")
    # Apply your tagger on dev & test data
    auto_dev_data = model.avg_tag(corpus.dev, args.epochs)
    auto_test_data = model.avg_tag(corpus.test, args.epochs)

    devfile = os.path.join(ap_dir, "dev.tagged")
    testfile = os.path.join(ap_dir, "test.tagged")
    # Outpur your auto tagged data
    output_auto_data(auto_dev_data, devfile)
    output_auto_data(auto_test_data, testfile)


parser = argparse.ArgumentParser()
parser.add_argument("-a", "--all", help="Run all", action="store_true")
parser.add_argument("--average", help="Average Perceptron", action="store_true")
parser.add_argument("--debug", help="Debug Mode", action="store_true")
parser.add_argument("-w", "--window", help="Window Size", type=int, default=5)
parser.add_argument("-e", "--epochs", help="Number of epochs", type=int, default=5)
parser.add_argument("-v", "--vocab", help="Percentage of vocabulary", type=float, default=1.)
parser.add_argument("-o", "--output", help="Output directory", type=str, default="auto")
parser.add_argument("-d", "--directory", help="Model Directory", type=str, default="models")
parser.add_argument("-r", "--remove", help="Remove Feature", type=str,
                    default=None, required=False)


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

    logger.info(f"Tags Count:")
    for k, v in label_dict.most_common(len(label_dict)):
        print(f"{k}: {v}")
        labels.append(k)

    corpus = Corpus(
        train=train_data, dev=gold_dev_data, test=test_data,
        labels=labels
    )

    if args.all:
        run_all(corpus, debug=args.debug, epochs=args.epochs,
                outdir=args.output, model_dir=args.directory)
    else:
        run(corpus=corpus, remove_feature=args.remove, epochs=args.epochs,
            outdir=args.output, model_dir=args.directory)
