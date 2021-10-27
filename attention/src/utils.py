#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Utility classes and functions"""
import json
import os
import random
import time
from argparse import Namespace
from typing import Iterable, List, Optional, Tuple, Union

import sacrebleu
import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vectors

import consts as C


##################################### PTB ######################################
class PTB(Dataset):
    """Penn TreeBank Corpus"""

    def __init__(self, name: str, data: C.DATASET_T, sent_vocab: C.VOCAB_T,
                 tree_vocab: C.VOCAB_T):
        """PTB init fn

        Args:
          name: name of dataset: dev, test or train
          data: raw ptb as a tuple of sents and trees
          sent_vocab: Tuple of sent_itos and sent_stoi
          tree_vocab: Tuple of tree_itos and tree_stoi
        """
        print(f"\n{name.capitalize()} PTB init")
        self.name = name

        self._raw_data = data
        self.sent_itos, self.sent_stoi = sent_vocab
        self.tree_itos, self.tree_stoi = tree_vocab

        self._data = []
        self._vectorize()

    def _vectorize(self):
        """converts raw string into indices using vocabs

        Source sentences are appended with EOS at the end. Target trees are appended
        with BOS and EOS at the beginning and end, respectively.
        """
        vectors = []
        for sent, tree in zip(*self._raw_data):
            sent_idx = convert_seq(sent.split() + [C.EOS], self.sent_stoi)
            tree_idx = convert_seq(
                [C.BOS] + tree.split() + [C.EOS], self.tree_stoi, is_target=True)
            vectors.append((sent_idx, tree_idx))

        # sample vector
        sample_idx = random.randint(0, len(vectors) - 1)
        print("Sample vector from", self.name)
        print("  Sent:", self._raw_data[0][sample_idx])
        print("  Sent Vector:", vectors[sample_idx][0])
        print("  Tree:", self._raw_data[1][sample_idx])
        print("  Tree Vector:", vectors[sample_idx][1])

        self._data = vectors

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sent, tree = self._data[idx]

        # into Tensor
        sent = torch.tensor(sent, dtype=torch.long)
        tree = torch.tensor(tree, dtype=torch.long)

        return sent, tree

    def __len__(self):
        return len(self._data)


#################################### VOCAB #####################################
def convert_seq(seq: Union[Iterable[str], Iterable[int], str], vocab: Union[C.ITOS_T, C.STOI_T],
                is_target=False, return_str=False) -> Union[Iterable[Union[str, int]], str]:
    """converts an incoming sequence into its mapping based on vocab

    When the the sequence is target, will raise exception when an unknown token
    is encountered.

    Args:
      seq: list ints or strs, or possiby a str to be converted
      vocab: stoi (list) or itos (dict)
      is_target: whether `seq` is a target sequence
      return_str: whether to return the output as a string

    Returns:
      If return_str is true: a string delimited by white-space
      Otherwise: a list of either str or int
    """
    if type(seq) is str:
        seq = seq.split()

    out_seq = []
    for tok in seq:
        try:
            out_seq.append(vocab[tok])
        except KeyError:
            if is_target:
                raise RuntimeError(f"Unknown target token: `{repr(tok)}` from vocab: {', '.join(vocab)}")
            out_seq.append(vocab[C.UNK])

    if return_str:
        out_seq = " ".join(out_seq)

    return out_seq


def validate_vocab(vocab: C.STOI_T, is_target=False):
    """validates the integrity of vocab data

    Args:
      vocab: stoi dict to be validated
      is_target: whether `vocab` is that of target

    Raises:
      AssertionError for following cases:
        1. PAD does not exist in vocab at index 0
        2, EOS does not exist in vocab
        3. UNK does not exist in source vocab
        4. EOS does not exist in target vocab
        5. Empty string exists in vocab
    """
    assert vocab[C.PAD] == 0

    assert C.EOS in vocab
    if is_target:
        assert C.BOS in vocab
    else:
        assert C.UNK in vocab

    assert "" not in vocab  # disallow empty string


#################################### MODEL #####################################
def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Counts the number of parameters in a given model whose requires_grad
    attribute is True

    Args:
      model: model to be inspected

    Returns:
      total number of parameters as int
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def raw_corpus_bleu(hypotheses: Iterable[str], references: Iterable[str],
                    offset: Optional[float] = 0.01) -> float:
    """
    Simple wrapper around sacreBLEU's BLEU without tokenization and smoothing.

    from https://github.com/awslabs/sockeye/blob/master/sockeye/evaluate.py#L37

    :param hypotheses: Hypotheses stream.
    :param references: Reference stream.
    :param offset: Smoothing constant.
    :return: BLEU score as float between 0 and 1.
    """
    return sacrebleu.raw_corpus_bleu(
        hypotheses, [references], smooth_value=offset).score


##################################### MISC #####################################
def display_exec_time(begin: float, msg_prefix: str = ""):
    """Displays the script's execution time

    Args:
      begin (float): time stamp for beginning of execution
      msg_prefix (str): display message prefix
    """
    exec_time = time.time() - begin

    msg_header = "Execution Time:"
    if msg_prefix:
        msg_header = msg_prefix.rstrip() + " " + msg_header

    if exec_time > 60:
        et_m, et_s = int(exec_time / 60), int(exec_time % 60)
        print("\n%s %dm %ds" % (msg_header, et_m, et_s))
    else:
        print("\n%s %.2fs" % (msg_header, exec_time))


def display_args(args: Namespace):
    """displays all flags

    Args:
      args: command line arguments
    """
    print("***** FLAGS *****")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")


def locate_state_dict_file(dir_path: str) -> Tuple[Optional[str], int]:
    """attempts to find a state_dict model parameter file within the given directory

    Args:
      dir_path: parent directory to search

    Returns:
      Tuple of located state_dict file (None if not found) and the BLEU score at
        its checkpoint
    """
    out = None

    best_bleu = -1
    for file in os.listdir(dir_path):
        filename, ext = os.path.splitext(file)
        if ext == '.pt':
            bleu_score = float(filename.split('_')[-1])
            if bleu_score > best_bleu:
                best_bleu = bleu_score

    if best_bleu > -1:
        out = C.MODEL_PT_FORMATTER.format(best_bleu)

    return out, best_bleu


def maybe_update_args(args: Namespace, other_args_dict: dict, ignore_flags=None) -> bool:
    """compares two commandline arguments and returns whether any important
    hyperparameter value has been updated in the current args (`args`) from the
    loaded args (`other_args_dict`) as a dict.

    `Important` hyperparameter is any that is not in IGNORE_FLAGS in consts.py

    As a side effect, hyperparameters that are updated in `args` will override
    those from `other_args_dict`

    Args:
      args: Current args Namespace
      other_args_dict: Previous args Namespace as a dict
      ignore_flags: optional list of flags to ignore

    Returns:
      whether any important hyperparameters are different in value
    """
    is_different = False

    if not ignore_flags:
        ignore_flags = C.IGNORE_FLAGS

    args_dict = vars(args)
    for k, v in other_args_dict.items():
        if k in args_dict:
            cur_v = args_dict[k]

            if 'dir' in k:
                try:
                    v = os.path.abspath(v)
                    cur_v = os.path.abspath(cur_v)
                except TypeError:
                    pass

            if v != cur_v:
                print(f"[!] Overriding `{k}` from {v} to {cur_v}")
                args.__dict__[k] = cur_v
                if k not in ignore_flags:
                    is_different = True
            else:
                args.__dict__[k] = v

    return is_different


def set_seed(seed: int):
    """for replicability"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def to_device(data: Iterable[torch.tensor], device: torch.device) -> Iterable[torch.tensor]:
    """moves incoming data to the device"""
    return [x.to(device) for x in data]


def use_gpu(debug=False) -> bool:
    """whether to use GPU if possible

    Args:
      debug: whether debugging mode is on

    Returns:
      True if GPU available and not debugging mode, otherwise False
    """
    return torch.cuda.is_available() and not debug


#################################### EXPORT ####################################
def export_json(obj: dict, path: str):
    """exports a json obj at path"""
    assert type(obj) is dict

    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)


def export_txt(obj: List[str], path: str, delimiter='\n'):
    """exports a txt obj joined by delimiter at path"""
    if type(obj) is list:
        obj = delimiter.join(obj)

    assert type(obj) is str

    with open(path, 'w') as f:
        f.write(obj)


def export_vocabs(vocabs: C.VOCABS_T, out_dir: str):
    """exports itos and stoi separately within out_dir"""
    sent_vocab, tree_vocab = vocabs

    sent_vocab_path = os.path.join(out_dir, C.SENT_VOCAB_FILENAME)
    export_txt(sent_vocab[0], sent_vocab_path)

    tree_vocab_path = os.path.join(out_dir, C.TREE_VOCAB_FILENAME)
    export_txt(tree_vocab[0], tree_vocab_path)


def export_ptb_dataset(dataset: C.DATASET_T, out_dir: str, name: str):
    """exports raw ptb dataset within out_dir"""
    assert name in C.PTB_SPLITS  # sentinel

    sents, trees = dataset

    sent_file = os.path.join(out_dir, C.SENT_DATA_FORMATTER.format(name))
    export_txt(sents, sent_file)

    tree_file = os.path.join(out_dir, C.TREE_DATA_FORMATTER.format(name))
    export_txt(trees, tree_file)


##################################### LOAD #####################################
def load_glove(glove_dir: str, embed_dim=50) -> Vectors:
    """loads a glove pre-trained embedding from glove_dir, further specified by
    embed_dim

    4 special symbols are inserted at the beginning of the matrix.
    1. PAD init as zero vector, placed at index 0
    2. UNK init as a mean across all GloVe vectors, placed at index 1
    3. BOS init as a random vector, placed at index 2
    4. EOS init as a random vector, placed at index 3

    Args:
      glove_dir: path to glove dir
      embed_dim: which embedding size to load

    Returns:
      GloVe as torchtext Vectors obj
    """
    print("Loading Glove")
    glove_name = os.path.basename(os.path.normpath(glove_dir))
    glove_filename = f'{glove_name}.{embed_dim}d.txt'
    glove = Vectors(glove_filename, cache=glove_dir)

    for symbol in C.SPECIAL_SYMBOLS:
        assert symbol not in glove.itos

    # UNK as a mean of all existing vectors
    glove.itos.insert(0, C.PAD)
    glove.itos.insert(1, C.UNK)
    glove.itos.insert(2, C.BOS)
    glove.itos.insert(3, C.EOS)

    glove.stoi[C.PAD] = 0
    glove.stoi[C.UNK] = 1
    glove.stoi[C.BOS] = 2
    glove.stoi[C.EOS] = 3

    pad_tensor = torch.zeros(1, embed_dim)  # zero padding vector
    unk_tensor = torch.mean(glove.vectors, axis=0).unsqueeze(0)
    bos_tensor = torch.rand(1, embed_dim)  # random vector for <bos>
    eos_tensor = torch.rand(1, embed_dim)  # random vector for <eos>
    glove.vectors = torch.cat(
        [pad_tensor, unk_tensor, bos_tensor, eos_tensor, glove.vectors], axis=0)

    validate_vocab(glove.stoi)

    r, c = glove.vectors.shape
    print(f"Successfully loaded => ({r} by {c})")
    return glove


def load_json(path) -> dict:
    """loads a json dict obj from path"""
    assert os.path.exists(path)
    with open(path, 'r') as f:
        out = json.load(f)
    return out


def load_txt(path, delimiter='\n') -> List[str]:
    """loads a txt obj from path, which is subsequently split by delimiter"""
    assert os.path.exists(path)
    with open(path, 'r') as f:
        out = f.read().split(delimiter)
    return out


def load_vocab(vocab_path: str) -> Tuple[C.ITOS_T, C.STOI_T]:
    """loads a single vocab object (stoi or itos) from vocab_path"""
    itos = load_txt(vocab_path)
    stoi = {word: i for i, word in enumerate(itos)}
    return itos, stoi


def load_vocabs(data_dir) -> Tuple[C.VOCAB_T, C.VOCAB_T]:
    """loads itos and stoi from data_dir"""
    sent_vocab_path = os.path.join(data_dir, C.SENT_VOCAB_FILENAME)
    assert os.path.exists(sent_vocab_path)
    sent_vocab = load_vocab(sent_vocab_path)
    tree_vocab_path = os.path.join(data_dir, C.TREE_VOCAB_FILENAME)
    assert os.path.exists(tree_vocab_path)
    tree_vocab = load_vocab(tree_vocab_path)
    return sent_vocab, tree_vocab


def load_ptb_dataset(data_dir, dataset_name: str) -> C.DATASET_T:
    """loads a raw ptb dataset from data_dir"""
    sent_path = os.path.join(data_dir, C.SENT_DATA_FORMATTER.format(dataset_name))
    assert os.path.exists(sent_path)
    sents = load_txt(sent_path)

    tree_path = os.path.join(data_dir, C.TREE_DATA_FORMATTER.format(dataset_name))
    assert os.path.exists(tree_path)
    trees = load_txt(tree_path)

    return sents, trees
