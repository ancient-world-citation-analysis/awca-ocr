"""Contains typical NLP utility functions that are designed to capture
our concerns.
"""

from typing import Callable, Dict, Hashable, List, Sequence, Tuple, TypeVar, \
    cast
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
    n: int
) -> Callable[[Sequence[Unit]], Frequencies]:
    """Returns a function that returns the counts of all n-grams in a
    sequence.
    """
    def count(units: Sequence[Unit]) -> Frequencies:
        ret: Frequencies = dict()
        for i in range(len(units) - n):
            key = cast(Tuple[Unit], tuple(units[i:i+n]))
            ret[key] = ret.get(key, 0) + 1
        return ret
    return count


def multiset_jaccard_similarity(
    truth: Frequencies,
    check: Frequencies
) -> float:
    return 1 - sum(
        abs(truth.get(key, 0) - check.get(key, 0))
        for key in truth.keys() | check.keys()
    ) / sum(
        truth[key] for key in truth
    )


TOKENIZER: Tokenizer = lambda s: [
    w for w in re.split(r'(\w+)|(?=\W)|(?<=\W)', s) if w and w.strip()
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
TRIGRAM = n_gram_counter(3)
MONOGRAM = n_gram_counter(1)
