from typing import Sequence
from typing import Callable, Optional, List, Tuple, Dict
from textprobability.classify import Classifier, default_classifier

"""This module contains language detection tools built on top of CLD3.
These are convenience functions for special use cases that CLD3 does
not explicitly address.
"""

LinguisticUnit = str
LangCode = str
LanguageAnnotator = Callable[
    [Sequence[LinguisticUnit]],
    List[LangCode]
]


def get_language_annotator(
    n_chars: Optional[int] = 100,
    window_size: Optional[int] = None,
    classifier: Classifier = default_classifier
) -> LanguageAnnotator:
    """Returns a LanguageAnnotator.

    A LanguageAnnotator is a function that annotates sequences of
    linguistic units with their respective languages. It does this by
    returning a list of BCP-47-style language codes.

    The user may decide what linguistic units to use. Suggested units
    include tokens, sentences, paragraphs, or proxy values thereof (such
    as sequences of five characters as a proxy for English tokens).
    """
    assert (
        (not n_chars and window_size)
        or (n_chars and not window_size)
    ), 'min_n_chars xor window_size must be a nonzero integer.'

    ResultSelection = Dict[LangCode, float]
    Index = int

    def get_window(
        s: List[LinguisticUnit], start_idx: Index
    ) -> Tuple[str, Index]:
        """Returns a concatenation of linguistic units w starting at
        `start_idx`.
        Returns the index of the linguistic unit that immediately
        follows the final linguistic unit included in w.
        """
        if window_size:
            end_idx = min(len(s), start_idx + window_size)
            return ' '.join(s[start_idx:end_idx]), end_idx
        else:
            ret = list()
            total_len = 0
            idx = start_idx
            while total_len < n_chars and idx < len(s):
                ret.append(s[idx])
                total_len += len(s[idx])
                idx += 1
            return ' '.join(ret), idx

    def get_votes(
        votes: List[ResultSelection],
        s: List[LinguisticUnit],
        contested_unit_indices: List[Index],
        update_start_idx: Callable[[Index, Index], Index],
        get_window: Callable[[List[LinguisticUnit], Index], Tuple[str, Index]]
    ):
        """Updates the weights associated with the various languages
        that might be associated with the linguistic units.
        """
        start_idx = 0
        while start_idx != len(s):
            window, end_idx = get_window(s, start_idx)
            if any(
                start_idx <= idx < end_idx
                for idx in contested_unit_indices
            ):
                for language, weight in classifier(window).items():
                    for idx in range(start_idx, end_idx):
                        votes[idx][language] = (
                            votes[idx].get(language, 0)
                            + weight
                        )
            start_idx = update_start_idx(start_idx, end_idx)

    def contested_unit_indices(votes):
        """Returns the indices of contested linguistic units."""
        return [
            i for i in range(len(votes))
            if len([
                possibility for possibility in votes[i]
                if votes[i][possibility] > 0
            ]) > 0
        ]

    def language_annotator(s: Sequence[LinguisticUnit]) -> List[LangCode]:
        s = list(s)
        votes = [dict() for _ in range(len(s))]
        contested = list(range(len(s)))
        get_votes(votes, s, contested, lambda start, _: start + 1, get_window)
        return [
            max(results, key=lambda result: results[result])
            for results in votes
        ]

    return language_annotator
