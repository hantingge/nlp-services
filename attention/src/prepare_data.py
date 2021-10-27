#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Loads and preprocesses PTB dataset"""
import argparse
import logging
import os
import time
from collections import Counter
from typing import List

import nltk
import tqdm

import consts as C
import utils


argparser = argparse.ArgumentParser("PA4 Data Loading & Preprocessing Argparser")

# path flags
argparser.add_argument('--ptb_dir', default='./ptb',
                       help='path to ptb directory')
argparser.add_argument('--out_dir', default='./outputs/data',
                       help='path to data outputs')
# data preprocessing flags
argparser.add_argument('--lower', action='store_true',
                       help='whether to lower all sentence strings')
argparser.add_argument('--reverse_sent', action='store_true',
                       help='whether to reverse the source sentences')
argparser.add_argument('--prune', action='store_true',
                       help='whether to remove parenthesis for terminal POS tags')
argparser.add_argument('--XX_norm', action='store_true',
                       help='whether to normalize all POS tags to XX')
argparser.add_argument('--closing_tag', action='store_true',
                       help='whether to  attach closing tags')
# misc. flags
argparser.add_argument('--force', action='store_true',
                       help='whether to force running the entire script')


#################################### VOCAB #####################################
def compile_vocab(data: C.DATASET_T, most_common_n=-1) -> C.VOCABS_T:
    """compiles vocabs for sentences and linearized phrase structure trees separately.

    Args:
      data: Tuple of sentences and linearized trees
      most_common_n: how many most common vocabs from each vocab to keep. -1 to ignore.

    Returns:
      Tuple of sentence vocab and tree vocab
    """
    print("Compiling Vocab..")

    sent_counter = Counter()
    tree_counter = Counter()

    for sent, tree in zip(*data):
        sent_counter.update(sent.split())
        tree_counter.update(tree.split())

    def to_itos(counter):
        if most_common_n < 0:
            pruned_vocab = list(counter)
        else:
            pruned_vocab = counter.most_common(n=most_common_n)
            pruned_vocab = list(zip(*pruned_vocab))[0]

        return sorted(pruned_vocab, key=str, reverse=True)

    # source (sent) vocab
    sent_itos = to_itos(sent_counter)
    sent_itos.insert(0, C.PAD)  # PAD: 0
    sent_itos.insert(1, C.UNK)  # UNK: 1
    sent_itos.insert(2, C.BOS)  # BOS: 2
    sent_itos.insert(3, C.EOS)  # EOS: 3
    sent_stoi = {word: i for i, word in enumerate(sent_itos)}

    # target (tree) vocab
    tree_itos = to_itos(tree_counter)  # already includes <end>
    tree_itos.insert(0, C.PAD)  # PAD: 0
    tree_itos.insert(1, C.BOS)  # BOS: 1
    tree_itos.insert(2, C.EOS)  # EOS: 2
    tree_stoi = {word: i for i, word in enumerate(tree_itos)}

    utils.validate_vocab(sent_stoi)
    utils.validate_vocab(tree_stoi, is_target=True)

    sent_vocab = (sent_itos, sent_stoi)
    tree_vocab = (tree_itos, tree_stoi)

    return sent_vocab, tree_vocab


################################## NORMALIZE ###################################
def normalize_tok(tok: str, lower=False) -> str:
    """map certain raw tokens to normalized form

    modified from https://github.com/sloria/textblob-aptagger/blob/master/textblob_aptagger/taggers.py

    Args:
      tok: candidate token to be normalized
      lower: whether to lower-case

    Returns:
      normalized token as str
    """
    if tok.isdigit() and len(tok) == 4:
        return '!YEAR'
    elif tok[0].isdigit():
        return '!DIGITS'
    elif '*T*' in tok:
        return '*T*'
    elif lower:
        return tok.lower()
    else:
        return tok


def normalize_pos(tag: str, XX_norm=False) -> str:
    """normalization for POS tags in phrase structure trees

    Args:
      tag: candidate POS tag to be considered
      XX_norm: whether to normalize tag value to `XX`

    Returns:
      normalized POS tag as str
    """
    if XX_norm:
        return "XX"
    # e.g. (NP-SBJ (-NONE- *T*-1) ), paper doesn't mention any special treatment
    if '-' in tag:
        if tag == '-NONE-':
            return tag
        tag = tag[:tag.index('-')]  # drop trace index
    if '=' in tag:
        tag = tag[:tag.index("=")]
    return tag


################################### PROCESS ####################################
def linearize_parse_tree(tree: nltk.tree.Tree, prune=False, XX_norm=False,
                         closing_tag=False) -> str:
    """linearizes a phrase structure parse tree through a DFS while applying
    normalization to POS tags

    Args:
      tree: nltk tree object to be linearized
      prune: whether to reduce `(TAG1 (TAG2 ) )` to `(TAG1 TAG2 )`
      XX_norm: whether to normalize all POS tags to XX, i.e. `(TAG1 (TAG2 ..` to `(XX (XX ..`
      closing_tag: whether to append the corresponding POS tag after closing parenthesis,
        i.e. `(TAG1 TAG2 )` to `(TAG1 TAG2 )TAG1`

    Returns:
      linearized parse tree as str
    """
    tree = str(tree).strip().split()

    s, out = [], []
    for i, tok in enumerate(tree):
        if '(' in tok:
            idx = tok.index("(")
            tag = tok[idx + 1:]
            tag = normalize_pos(tag, XX_norm=XX_norm)
            s.append(tag)

            new_tok = '(' + tag
            out.append(new_tok)
        else:
            idx = tok.index(")")
            for _ in range(idx, len(tok)):
                new_tok = ')'
                tag = s.pop()
                if closing_tag:
                    new_tok += tag
                out.append(new_tok)

                if prune:
                    try:
                        prev_tok = out[-2]
                        if prev_tok.startswith('('):
                            prev_tok_tag = prev_tok[1:]
                            if prev_tok_tag == tag:
                                if XX_norm:
                                    tag = 'XX'
                                out = out[:-2] + [tag]
                    except IndexError:
                        pass

    out = list(filter(None, out))  # drop empty tokens
    return " ".join(out)


def process_sent(sent: List[str], lower=False, reverse=False) -> str:
    """processes a single sentence by applying token-level normalization

    Args:
      sent: sentence to be processed
      lower: whether to lower-case
      reverse: whether to reverse sentence

    Returns:
      processed sentence as str
    """
    if reverse:
        sent = reversed(sent)

    norm_sent = []
    for tok in sent:
        norm_tok = normalize_tok(tok, lower=lower)
        if norm_tok is not None:  # implicitly drops empty tokens
            norm_sent.append(norm_tok)

    return " ".join(norm_sent)


def process_sents(sents: List[C.SENTS_T], lower=False, reverse=False) -> C.SENTS_T:
    """processes multiple sentences

    Args:
      sents: sentences to be processed
      lower: see `process_sent`
      reverse: see `process_sent`

    Returns:
      List of processed sentences as List[str]
    """
    out_sents = []
    for sent in tqdm.tqdm(sents):
        out_sent = process_sent(sent, lower, reverse)
        out_sents.append(out_sent)

    return out_sents


def process_trees(trees: List[nltk.tree.Tree], prune=False, XX_norm=False,
                  closing_tag=False) -> C.TREES_T:
    """processes multiple trees

    Args:
      trees: List of nltk Tree objects to be processed
      prune: see `linearize_parse_tree`
      XX_norm: see `linearize_parse_tree`
      closing_tag: see `linearize_parse_tree`

    Returns:
      List of processed trees as List[str]
    """
    out_trees = []
    for tree in tqdm.tqdm(trees):
        tree = linearize_parse_tree(
            tree, prune=prune, XX_norm=XX_norm, closing_tag=closing_tag)
        out_trees.append(tree)
    return out_trees


def load_ptb_dataset(data_dir: str, dataset_name: str, lower=False, reverse=False,
                     prune=False, XX_norm=False, closing_tag=False) -> C.DATASET_T:
    """Loads a single PTB dataset (dev, test or train)

    Args:
      data_dir: PTB home directory
      dataset_name: name of dataset to load (dev, test or train)
      lower: see `process_sent`
      reverse: see `process_sent`
      prune: see `linearize_parse_tree`
      XX_norm: see `linearize_parse_tree`
      closing_tag: see `linearize_parse_tree`

    Returns:
      Tuple of processed sents and trees
    """
    assert dataset_name in C.PTB_SPLITS  # sentinel

    print(f"\nLoading {dataset_name}..")
    dataset_dir = os.path.join(data_dir, dataset_name)

    reader = nltk.corpus.BracketParseCorpusReader(dataset_dir, r'.*/wsj_.*\.mrg')

    print(f"Loading and processing sentences")
    sents = process_sents(reader.sents(), lower=lower, reverse=reverse)

    print(f"Loading and processing parse trees")
    trees = process_trees(
        reader.parsed_sents(), prune=prune, XX_norm=XX_norm, closing_tag=closing_tag)

    print("Sample data from", dataset_name)
    print("  Sent:", sents[0])
    print("  Tree:", trees[0])

    return sents, trees


def prepare_data(args: argparse.Namespace):
    """main data prep function"""
    # setup
    assert os.path.exists(args.ptb_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    print("\nBegin loading and processing PTB..")

    # load raw ptb data for dev, test and train
    for dataset_name in C.PTB_SPLITS:
        ptb_dataset = load_ptb_dataset(
            args.ptb_dir, dataset_name,
            lower=args.lower,
            reverse=args.reverse_sent,
            prune=args.prune,
            XX_norm=args.XX_norm,
            closing_tag=args.closing_tag)

        # export processed dataset
        utils.export_ptb_dataset(ptb_dataset, args.out_dir, dataset_name)

        # compile and export vocab from train
        if dataset_name == 'train':
            vocabs = compile_vocab(ptb_dataset)
            utils.export_vocabs(vocabs, args.out_dir)


def check_all_data_exist(data_dir):
    flags = []

    def exists(filename):
        return os.path.exists(os.path.join(data_dir, filename))

    # check PTB  exist
    for dataset_name in C.PTB_SPLITS:
        flags.append(exists(C.SENT_DATA_FORMATTER.format(dataset_name)))
        flags.append(exists(C.TREE_DATA_FORMATTER.format(dataset_name)))

    # check vocabs exist
    flags.append(exists(C.SENT_VOCAB_FILENAME))
    flags.append(exists(C.TREE_VOCAB_FILENAME))

    return all(flags)


def main():
    """Data prep script entry point"""
    begin = time.time()

    args, _ = argparser.parse_known_args()
    os.makedirs(args.out_dir, exist_ok=True)
    args_path = os.path.join(args.out_dir, C.ARGS_JSON_FILENAME)

    is_different = False
    if os.path.exists(args_path):
        print("Loading args from", args_path)
        prev_args_dict = utils.load_json(args_path)
        is_different = utils.maybe_update_args(args, prev_args_dict, ignore_flags=C.DATA_IGNORE_FLAGS)

    # re-run if configs has changed, or ptb files dont exist
    load_data = is_different or not check_all_data_exist(args.out_dir)

    utils.display_args(args)

    if load_data or args.force:
        utils.export_json(vars(args), args_path)
        prepare_data(args)
    else:
        print(f"\nCurrent data in {args.out_dir} already satisfies given configurations.")
        print("Returning..")

    utils.display_exec_time(begin, "PA4 Data Loading & Preprocessing")


if __name__ == '__main__':
    
    main()
