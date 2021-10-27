import copy
import os
from functools import lru_cache
import time
from collections import defaultdict, deque, Counter
from typing import List, Dict, Tuple, Sequence, Set, Optional
from math import log
from operator import itemgetter
import random
import pickle


import numpy as np
from tqdm import tqdm
from numba import njit, jit, prange
from numba import types
from numba.typed import Dict as NumbaDict, List as NumbaList

from data_structures import Sentence, ItemFeatures

__author__ = "Hanting Ge"

START_TAG = "START"
OOV_TOKEN = "<unk>"


# Use these functions in your Bigram Taggers
def _safe_log(n: float) -> float:
    """Return the natural log of n or -inf if n is 0.0."""
    return float("-inf") if n == 0.0 else log(n)


def _max_item(scores: Dict[str, float]) -> Tuple[str, float]:
    """Given a dict of tag: score, return a tuple of the max tag and its score."""
    return max(scores.items(), key=itemgetter(1))


def _accuracy(gold_lines, auto_lines) -> float:
    correct = 0.0
    total = 0.0
    for g_snt, a_snt in zip(gold_lines, auto_lines):
        correct += sum([g_tup == a_tup for g_tup, a_tup in zip(g_snt, a_snt)])
        total += len(g_snt)

    return correct / total


def cal_speed(start: float) -> float:
    return time.time() - start


class Perceptron_POS_Tagger(object):
    def __init__(self, labels: Optional[List[str]], output=None):
        self.tags = labels
        self.params: Dict[Tuple[str, str], float] = defaultdict(float)
        self.epoch = 0
        self.output = output
        self.history = []

    def save(self, loc: str):
        """Saves model to loc"""
        if not (os.path.exists(loc) or os.path.isdir(loc)):
            os.mkdir(loc)
        state_dict = {
            "epoch": self.epoch, "params": self.params,
            "tags": self.tags, "history": self.history
        }
        with open(os.path.join(loc, "model.pkl"), 'wb') as f:
            pickle.dump(state_dict, f)

    @classmethod
    def from_disk(cls, loc: str) -> "Perceptron_POS_Tagger":
        model = cls(labels=None)
        try:
            with open(loc, 'rb') as f:
                params = pickle.load(f)
        except Exception:
            raise FileNotFoundError(f"{loc} not found")
        model.params = params["params"]
        model.tags = params["tags"]
        model.epoch = params["epoch"]
        try:
            model.history = params["history"]
        except KeyError:
            print("History was not saved")
        return model

    def _from_disk(self, loc: str):
        try:
            with open(loc, 'rb') as f:
                params = pickle.load(f)
        except Exception:
            raise FileNotFoundError(f"{loc} not found")
        self.params = params["params"]
        self.tags = params["tags"]
        self.epoch = params["epoch"]
        try:
            self.history = params["history"]
        except KeyError:
            print("History was not saved")

    def evaluate(self, eval_data: List[Sentence]) -> float:
        """Evaluates tagger on  dev set"""
        preds, golds = [], []

        for sent in tqdm(eval_data, desc="Eval"):
            pred = self.viterbi(sent.features)
            preds.append(pred)
            golds.append(sent.tags)

        return round(_accuracy(golds, preds), 5)

    def tag(self, test_data: Sequence[Sentence]) -> Sequence[Sequence[str]]:
        """Predicts POS tags on test data"""
        if self.output is not None:
            self._from_disk(self.output)  # Load best parameters
        res = []
        for sent in tqdm(test_data, desc="Decoding"):
            pred = []
            features = sent.features
            tags = self.viterbi(features)
            for word, tag in zip(sent.words, tags):
                pred.append("_".join([word, tag]))
            res.append(pred)
        return res

    def avg_tag(self, test_data: Sequence[Sentence], epochs: int):
        """Averaged perceptron"""
        if self.output is not None:
            self._from_disk(self.output)  # Load best parameters
        avg_params = defaultdict(float)
        for f, v in self.params.items():
            avg_params[f] = v / epochs
        params = copy.deepcopy(self.params)
        self.params = avg_params
        res = self.tag(test_data)
        self.params = params  # Switch back to original parameters
        return res

    def train(
            self, train_data: List[Sentence], dev_data: List[Sentence], 
            epochs: int = 10, eta: float = 1e-2, batch_size: int = 128,
            random_state: int = 1234, use_prev_tag: bool = True
    ):
        """Trains POS tagger"""
        print(f"Use Prev Tag: {use_prev_tag}")
        random.seed(random_state)
        eval_data = random.sample(dev_data, len(dev_data)//8)

        total = len(train_data)
        best_acc = float("-inf")
        for self.epoch in range(epochs):
            random.shuffle(train_data)
            batches = [
                train_data[idx:idx+batch_size] for idx in range(
                    0, len(train_data), batch_size
                )
            ]
            start = time.time()

            for batch in tqdm(batches, desc=f"epoch {self.epoch+1}/{epochs}"):
                labels_pred, labels_gold, batch_features = [], [], []
                for s in batch:
                    pred = self.viterbi(s.features)
                    labels_pred.extend(pred)
                    labels_gold.extend(s.tags)
                    batch_features.extend(s.features)

                assert len(batch_features) == len(labels_gold) == len(labels_pred)

                self.update(
                    batch_features, labels_gold, labels_pred, eta, use_prev_tag
                )
            speed = round(cal_speed(start) / total, 4)
            acc = self.evaluate(eval_data)
            self.history.append(acc)
            if self.output is not None and acc > best_acc:
                self.save(self.output)
                bset_acc = acc
            print(f"epoch {self.epoch+1}/{epochs}, "
                  f"dev_acc={acc}, "
                  f"speed={speed} secs/sent")

    def update(
            self, local_features,
            labels_true, labels_pred,
            eta: float, use_prev_tag: bool = True
    ) -> None:
        """Updates feature weights"""
        for i, (label_true, label_pred, features) in enumerate(
                zip(labels_true, labels_pred, local_features)
        ):
            # Update init weights
            if i == 0:
                self.params[(START_TAG, label_true)] += eta
                self.params[(START_TAG, label_pred)] -= eta
            # Updates transition weights
            if use_prev_tag and i > 0:
                # Increase weights for gold features + labels
                self.params[(labels_true[i-1], label_true)] += eta
                # Decrease weights for auto features + labels
                self.params[(labels_pred[i-1], label_pred)] -= eta
                # Note: if label_pred == label_true, no change in weights
                # In other words, if prediction is correct, no change
                # If prediction is wrong, add 1 for gold and minus 1 for pred feature

            # Updates emission weights
            for feature in features:
                self.params[(feature, label_true)] += eta
                self.params[(feature, label_pred)] -= eta

    def extract_global_features(
            self, labels_true, labels_pred, local_features
    ):
        """Extracts global features given a sequence of labels/tags
        and a sequence of local features(features for each token)
        """
        features_pred = defaultdict(float)
        features_gold = defaultdict(float)
        # Iterate thru each label and feature
        for i, (label_true, label_pred, features) in enumerate(
                zip(labels_true, labels_pred, local_features)
        ):
            # Add transition weights
            if i == 0:
                features_pred[('START', label_pred)] += 1.
                features_gold[('START', label_true)] += 1.
            else:
                features_pred[(labels_pred[i-1], labels_pred[i])] += 1.
                features_gold[(labels_true[i-1], labels_true[i])] += 1.
            # Add emission weights
            for feature, weight in features.items():
                features_pred[(feature, labels_pred[i])] += weight
                features_gold[(feature, labels_true[i])] += weight
        return features_gold, features_pred

    def _viterbi(self, sentence: Sequence[ItemFeatures]) -> Sequence[str]:
        """Uses Viterbi algorithm to tag sentence
        Time: O(N*C^2)
        Hashmap implementation instead of matrix
        """
        N = len(sentence)
        viterbi: List[Dict[str, float]] = [
            {tag: float("-inf") for tag in self.tags} for _ in range(N)
        ]

        backpointers = [
            {tag: -1 for tag in self.tags} for _ in range(N)
        ]

        # Initialize start probs
        for tag in self.tags:
            viterbi[0][tag] = sum(
                [self.params[(x, tag)] for x, w in sentence[0].items()]
            ) + self.params[('START', tag)]

        # Viterbi algorithm, compute sequence prob

        for r in range(1, N):
            for curr_tag in self.tags:
                best_score = float("-inf")
                for prev_tag in self.tags:
                    prev_prob = viterbi[r-1][prev_tag]
                    emission = sum(
                        [
                            self.params[(x, curr_tag)] for x, _ in sentence[r].items()
                        ]
                    )
                    transition = self.params[(prev_tag, curr_tag)]
                    prob = prev_prob + emission + transition
                    if prob > best_score:
                        best_score = prob
                        viterbi[r][curr_tag] = prob
                        backpointers[r][curr_tag] = prev_tag

        best_tag, best_prob = _max_item(viterbi[N-1])
        assert viterbi[N - 1][best_tag] == best_prob
        best_path = [best_tag]

        for r in range(N-1, 0, -1):
            tag = backpointers[r][best_tag]
            best_path.insert(0, tag)
            best_tag = tag

        return best_path

    def viterbi(self, features: Sequence[ItemFeatures]) -> Sequence[str]:
        """Implements Viterbi decoding to recognize part of speech
        Matrix implementation
        """
        N = len(self.tags)  # Number of states
        T = len(features)  # Number of observations

        # Initialize Viterbi and backpointers matrix
        # Row represents position in sentence
        # Col represents states
        viterbi = [[-np.inf for _ in range(N)] for _ in range(T)]
        backpointers = [[-1 for _ in range(N)] for _ in range(T)]

        # Initialize the viterbi matrix with starting probability
        for curr in range(N):  # Current hidden state
            #transition = self.params[(START_TAG, self.tags[curr])]
            transition = self.dump_transition(START_TAG, self.tags[curr])
            emission = self.dump_emission(features[0], self.tags[curr])
            viterbi[0][curr] = transition + emission

        # Update viterbi scores
        # Iterate thru every token in sentence
        for t in range(1, T):  # t is observable state
            # Iterate thru every state
            for curr in range(N):  # curr is current hidden state
                best_score = -np.inf

                emission = self.dump_emission(features[t], self.tags[curr])
                # Find maximum path weight (previous viterbi + transition + current emission)
                for prev in range(N):  # prev is previous hidden state
                    v = viterbi[t-1][prev]  # Previous viterbi score
                    transition = self.dump_transition(
                        self.tags[prev], self.tags[curr]
                    )  # transition
                    score = v + transition + emission

                    if score > best_score:
                        best_score = score
                        viterbi[t][curr] = score
                        backpointers[t][curr] = prev
                assert backpointers[t][curr] != -1

        # Get last row of viterbi matrix
        last_row = np.array(viterbi[T-1]).reshape((-1, 1))
        path = []
        best_prob = last_row.max(axis=0)
        max_index = last_row.argmax()

        assert viterbi[T-1][max_index] == best_prob

        path.insert(0, self.tags[max_index])
        for i in range(T-1, 0, -1):
            index = backpointers[i][max_index]
            path.insert(0, self.tags[index])
            max_index = index

        return path

    def dump_emission(self, features: ItemFeatures, tag: str) -> float:
        """Computes emission weight
        Time: O(N)
        """
        return sum([self.params[(f, tag)] for f in features])


    def dump_transition(self, prev_tag: str, curr_tag: str) -> float:
        """Computes transition weight"""
        return self.params[(prev_tag, curr_tag)]



