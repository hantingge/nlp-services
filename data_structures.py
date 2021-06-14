import math
from collections import Counter
import logging
from typing import Mapping, Sequence, Tuple, Any, List, Set

from numba.typed import List as NumbaList

ItemFeatures = Mapping[str, float]
OOV = "<unk>"

logger = logging.getLogger(__name__)
__author__ = "Hanting Ge"


class Token:
    """Token class"""
    def __init__(self, token: str, tag: str):
        self.text = token
        self.tag = tag

    def __str__(self):
        return f"{self.text}_{self.tag}"

    def __repr__(self):
        return f"<Token {str(self)}>"

    def __eq__(self, other: Any):
        return (
            isinstance(other, Token)
            and self.text == other.text
            and self.tag == other.tag
        )

    def __lt__(self, other: "Token"):
        return self.to_tuple() < other.to_tuple()

    def __hash__(self):
        return hash(self.to_tuple())

    def to_tuple(self):
        """ Convert an instance of Token to a tuple of the token's text and its tag.
        Example:
            >>> token = Token("apple", "NN")
            >>> token.to_tuple()
            ("apple", "NN")
        """
        return self.text, self.tag

    @staticmethod
    def from_tuple(t: Tuple[str, ...]):
        """Create a Token object from a tuple. """
        assert len(t) == 2
        return Token(t[0], t[1])

    @staticmethod
    def from_string(s: str) -> "Token":
        """Create a Token object from a string with the format 'token/tag'.

        Usage: Token.from_string("cat_NN")
        """
        return Token(*s.rsplit("_", 1))

    def prefix(self, len: int):
        return self.text[:len]

    def suffix(self, len: int):
        return self.text[-len:]


class Sentence(object):
    """The sentence class"""
    def __init__(self, snt: Sequence[Any], use_numba: bool = False):
        if isinstance(snt[0], List):
            self.snt = [Token.from_tuple(t) for t in snt]
        else:
            self.snt = [Token.from_tuple((t, "UNK")) for t in snt]
        self.features = None
        if use_numba:
            self.features = self.extract_numba_features()
        else:
            self.features = self.extract_local_features()
        self.tags = self.get_tags()

    def extract_local_features(
            self, vocab=None, remove_feature=None
    ):
        """Extracts features at every index"""
        return [self.to_dict(
            self.snt, i, vocab, remove_feature
        ) for i in range(len(self.snt))]

    def extract_numba_features(self):
        return NumbaList([
            self.numba_features(self.snt, i) for i in range(len(self.snt))
        ])

    @staticmethod
    def get_text(sent: Sequence[Token], pos: int, vocab=None):
        """Returns the text of sent[pos]"""
        t = sent[pos].text.lower()
        return t if vocab is None or t in vocab else OOV

    def to_dict(
            self, sent: Sequence[Token], position: int, vocab=None,
            remove_feature=None
    ) -> ItemFeatures:
        """Extracts features with a window of 5"""
        features = {"bias": 1.0}
        if remove_feature != "curr":
            curr = self.get_text(sent, position, vocab)
            features[f"word={curr}"] = 1.0
            curr = sent[position].text
            if curr.istitle():
                features["titlecase"] = 1.0
            if curr.isupper():
                features["uppercase"] = 1.0
        if remove_feature != "prev":
            if position > 0:
                features[f"word-1={self.get_text(sent, position-1, vocab)}"] = 1.0
            if position > 1:
                features[f"word-2={self.get_text(sent, position-2, vocab)}"] = 1.0
        if remove_feature != "next":
            if position < len(sent) - 1:
                features[f"word+1={self.get_text(sent, position+1, vocab)}"] = 1.0
            if position < len(sent) - 2:
                features[f"word+2={self.get_text(sent, position+2, vocab)}"] = 1.0

        # For char_level features, we disregard oov
        if remove_feature != "ling":
            curr = sent[position].text.lower()
            prefix = curr[:3]
            suffix = curr[-3:]

            features[f"prefix={prefix}"] = 1.
            features[f"suffix={suffix}"] = 1.

        return features

    def numba_features(self, sent: Sequence[Token], position: int):
        """Extracts features with a window of 5"""
        return set(list(self.to_dict(sent, position)))

    def get_tags(self):
        return NumbaList([t.tag for t in self.snt])

    @property
    def words(self):
        return [t.text for t in self.snt]

    def __repr__(self):
        return " ".join(self.words)

    def __iter__(self):
        yield from self.snt

    def __len__(self):
        return len(self.snt)


class Corpus(object):
    def __init__(
            self, train, dev, test, labels, n: float = 1.
    ):
        self.train = train
        self.dev = dev
        self.test = test
        self.labels = labels
        self.vocab = self.build_vocab(n)
        self.featurize()

    def build_vocab(self, n: float) -> Set[str]:
        """Takes top n % vocab from train set
        Mask oov with unk"""
        counter = Counter()
        dev_vocab, test_vocab = set(), set()
        for s in self.train:
            counter.update(s.words)
        for s in self.dev:
            dev_vocab |= set(s.words)
        for s in self.test:
            test_vocab |= set(s.words)
        max_vocab = math.ceil(len(counter) * n)
        vocab = set()
        for v, _ in counter.most_common(max_vocab):
            vocab.add(v)
        logger.info(f"Vocab size: {len(vocab)}")
        logger.info(f"Dev Vocab size: {len(dev_vocab)}")
        logger.info(f"Test Vocab size: {len(test_vocab)}")
        logger.info(f"Dev OOV={round(len(dev_vocab-vocab)/len(dev_vocab)*100, 4)}%")
        logger.info(f"Test OOV={round(len(test_vocab-vocab)/len(test_vocab)*100, 4)}%")
        return vocab

    def featurize(
            self, remove_feature=None
    ) -> None:
        """Featurize sentences"""
        for i in range(len(self.train)):
            self.train[i].features = self.train[i].extract_local_features(
                self.vocab, remove_feature
            )
        for i in range(len(self.dev)):
            self.dev[i].features = self.dev[i].extract_local_features(
                self.vocab, remove_feature
            )
        for i in range(len(self.test)):
            self.test[i].features = self.test[i].extract_local_features(
                self.vocab, remove_feature
            )
