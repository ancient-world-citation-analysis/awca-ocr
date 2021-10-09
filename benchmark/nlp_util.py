"""Contains typical NLP utility functions that are designed to capture
our concerns.
"""

from typing import Callable, Dict, Hashable, Iterable, List, Sequence, Tuple, \
     TypeVar, cast
import re

Regex = str
Tokenizer = Callable[[str], List[str]]
Canonicalizer = Callable[[str], str]
Unit = TypeVar('Unit', bound=Hashable)
Frequencies = Dict[Tuple[Unit], int]


def get_canonicalizer(
    canonical: str,
    equivalents: Regex,
    compose: Canonicalizer = lambda x: x
) -> Canonicalizer:
    """Returns a canonicalizer -- a function that replaces matches to
    `equivalents` with `canonical`.
    """
    return lambda s: compose(re.sub(equivalents, canonical, s))


def n_gram_counter(
    n: int,
    valid: Callable[[Tuple[Unit]], bool]
) -> Callable[[Sequence[Unit]], Frequencies]:
    """Returns a function that returns the counts of all n-grams in a
    sequence.
    :param n: the number of units in an n-gram
    :param valid: a predicate on candidate n-grams such that for any
        tuple of Units x,
        (valid(x) = (the truth value of (x is an n-gram)))
    """
    def count(units: Sequence[Unit]) -> Frequencies:
        ret: Frequencies = dict()
        for i in range(len(units) - n + 1):
            key = cast(Tuple[Unit], tuple(units[i:i+n]))
            if valid(key):
                ret[key] = ret.get(key, 0) + 1
        return ret
    return count


def multiset_jaccard_similarity(
    truth: Frequencies,
    check: Frequencies
) -> float:
    """Returns the multiset Jaccard similarity of `check` to `truth`
    (which is asymmetric due to normalization by `truth`).
    """
    return 1 - sum(
        abs(truth.get(key, 0) - check.get(key, 0))
        for key in truth.keys() | check.keys()
    ) / sum(
        truth[key] for key in truth
    )


TOKENIZER: Tokenizer = lambda s: [
    w for w in re.split(r'(?=[^\w�])|(?<=[^\w�])', s) if w and w.strip()
]
CANONICALIZER: Canonicalizer = get_canonicalizer(
    '-', r'\-|\–',  # Hyphens and en dashes are the same
    get_canonicalizer(
        '—', r'\—|\―',  # Em dashes and horizontal bars are the same
        get_canonicalizer(
            '…', r'((\. ?){3,})|…',  # All kinds of ellipsis are the same
            get_canonicalizer(
                '\'', r'[\'‘’]',  # All kinds of single quote are the same
                get_canonicalizer(
                    '"', r'["“”]'  # All kinds of double quote are the same
                )
            )
        )
    )
)
_IS_NGRAM: Callable[[Tuple[Unit]], bool] = lambda x: '�' not in x and (
    not isinstance(x, Iterable) or not any(
        # This line is reached implies x is iterable
        '�' in sub for sub in cast(Iterable, x)
    )
)
TRIGRAM = n_gram_counter(3, _IS_NGRAM)
MONOGRAM = n_gram_counter(1, _IS_NGRAM)
