"""Contains typical NLP utility functions that are designed to capture
our concerns.
"""

from typing import Callable, Dict, Hashable, Iterable, List, Sequence, Tuple, \
     TypeVar, cast
import re

Regex = str
Unit = TypeVar('Unit', bound=Hashable)
Tokenizer = Callable[[str], List[Unit]]
Canonicalizer = Callable[[str], str]
NGramBag = Dict[Tuple[Unit], int]


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
) -> Callable[[Sequence[Unit]], NGramBag]:
    """Returns a function that returns the counts of all n-grams in a
    sequence.
    :param n: the number of units in an n-gram
    :param valid: a predicate on candidate n-grams such that for any
        tuple of Units x,
        (valid(x) = (the truth value of (x is an n-gram)))
    """
    def count(units: Sequence[Unit]) -> NGramBag:
        ret: NGramBag = dict()
        for i in range(len(units) - n + 1):
            key = cast(Tuple[Unit], tuple(units[i:i+n]))
            if valid(key):
                ret[key] = ret.get(key, 0) + 1
        return ret
    return count


def multiset_jaccard_similarity(a: NGramBag, b: NGramBag) -> float:
    """Returns the multiset Jaccard similarity of `a` and `b`.
    Multiple definitions of this metric seem to exist, but here is one
    version that has desirable properties:
    * Symmetry between the two operands
    * Penalty for failing to include a word that is in the other set
    * Penalty for including a word that the other set fails to include
    * Image is [0, 1], which aids communication
    * Consistency with set Jaccard distance, which also aids
    communication
    """
    keys = a.keys() | b.keys()
    try:
        return sum(
            min(a.get(key, 0), b.get(key, 0))
            for key in keys
        ) / sum(
            max(a.get(key, 0), b.get(key, 0))
            for key in keys
        )
    except ZeroDivisionError:
        # If both sets are empty, then they are as similar as
        # possible, and 1 is the maximum possible similarity.
        return 1


TOKENIZER: Tokenizer = lambda s: [
    w for w in re.split(r'(?<=[^\w�])|(?=[^\w�])', s) if w.strip()
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
SIMILARITY: Callable[[int], Callable] = lambda n: lambda a, b: (
    multiset_jaccard_similarity(
        n_gram_counter(n, _IS_NGRAM)(TOKENIZER(CANONICALIZER(a))),
        n_gram_counter(n, _IS_NGRAM)(TOKENIZER(CANONICALIZER(b)))
    )
)
