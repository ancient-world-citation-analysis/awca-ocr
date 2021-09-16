from typing import Sequence
import gcld3
from typing import Callable, Optional

"""This module contains language detection tools built on top of CLD3.
These are convenience functions for special use cases that CLD3 does
not explicitly address.
"""

# FIXME: Do not feed URLS, emails, etc. into gcld3.
DEFAULT_NNLI = gcld3.NNetLanguageIdentifier(1, 700)

LinguisticUnit = str
LangCode = str
LanguageAnnotator = Callable[
    [Sequence[LinguisticUnit]],
    list[LangCode]
]


def get_language_annotator(
    n_chars: Optional[int] = 100,
    window_size: Optional[int] = None,
    nnli: gcld3.NNetLanguageIdentifier = DEFAULT_NNLI,
    max_n_languages: int = 3
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

    ResultSelection = dict[LangCode, float]
    Index = int

    def get_window(
        s: list[LinguisticUnit], start_idx: Index
    ) -> tuple[str, Index]:
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

    def get_weight(result: gcld3.Result) -> float:
        """Returns the weight associated with a given language detection
        result.
        """
        return result.probability * result.proportion

    def get_votes(
        votes: list[ResultSelection],
        s: list[LinguisticUnit],
        contested_unit_indices: list[Index],
        update_start_idx: Callable[[Index, Index], Index],
        get_window: Callable[[list[LinguisticUnit], Index], tuple[str, Index]]
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
                for result in nnli.FindTopNMostFreqLangs(
                    window, max_n_languages
                ):
                    for idx in range(start_idx, end_idx):
                        votes[idx][result.language] = (
                            votes[idx].get(result.language, 0)
                            + get_weight(result)
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

    def language_annotator(s: Sequence[LinguisticUnit]) -> list[LangCode]:
        s = list(s)
        votes = [dict() for _ in range(len(s))]
        contested = list(range(len(s)))
        get_votes(votes, s, contested, lambda start, _: start + 1, get_window)
        return [
            max(results, key=lambda result: results[result])
            for results in votes
        ]

    return language_annotator
