"""Break text into passage-sized, page-sized, line-sized pieces."""

from typing import Any, Callable, List, Sequence
from numpy.random import Generator
from functools import reduce
import operator

def get_chunker(
    rng: Generator,
    min_line_len: int = 30,
    max_line_len: int = 60,
    max_page_len: int = 30,
    mean_section_len: float = 15000
) -> Callable[[str], str]:
    """Returns a text chunker that takes in text and produces pages of
    chunked and permuted text.
    :param rng: The random number generator that determines the behavior
        of the returned chunker
    :param max_line_len: The maximum number of characters per line
    :param mean_section_len: The mean number of characters per section of
        contiguous text
    """
    return composed(
        get_line_filter(min_line_len),
        get_section_breaker(rng, 1 / mean_section_len),
        get_permuter(rng),
        concatenator,
        get_line_breaker(max_line_len),
        get_page_breaker(max_page_len)
    )

def get_line_filter(min_line_len: int = 30) -> Callable[[str], str]:
    """Returns a function that filters out lines of text that are too
    short (where a line is defined to be a contiguous sequence of
    characters that are not the newline character '\n').
    :param min_line_len: The minimum line length required for a line of
        text to be included
    """
    def line_filter(original: str) -> str:
        out = list()
        next_line = list()
        for char in original + '\n':
            next_line.append(char)
            if char == '\n':
                if len(next_line) >= min_line_len:
                    out.extend(next_line)
                next_line = list()
        return ''.join(out)
    return line_filter

def get_line_breaker(max_line_len: int = 60) -> Callable[[str], str]:
    """Returns a function that inserts newlines as needed to ensure that
    no lines are longer than `max_line_len`.
    :param max_line_len: the maximum desired line length
    """
    def breaker(original: str) -> str:
        chars_since_newline = 0
        out = list()
        for char in original:
            out.append(char)
            if char != '\n':
                chars_since_newline += 1
            if chars_since_newline == max_line_len:
                out.append('\n')
            if not out or out[-1] == '\n':
                chars_since_newline = 0
        return ''.join(out)
    return breaker

def get_page_breaker(max_page_len: int = 30) -> Callable[[str], List[str]]:
    """Returns a function that splits strings into segments with only a
    limited number of lines.
    :param max_page_len: the maximum desired number of lines per page
    """
    def breaker(original: str) -> List[str]:
        ret = list()
        start = 0
        while True:
            break_position = start
            for _ in range(max_page_len):
                break_position = original.find('\n', break_position)
                if break_position == -1:
                    ret.append(original[start:])
                    return ret
                break_position += 1
            ret.append(original[start:break_position])
            start = break_position
    return breaker

def get_section_breaker(
    rng: Generator, r: float
) -> Callable[[Sequence[Any]], List[Sequence[Any]]]:
    """Returns a function that breaks a list into sublists with length
    distributed as Geometric(r).
    """
    def breaker(original: Sequence[Any]) -> List[Sequence[Any]]:
        break_indices = [0]
        for i in range(1, len(original)):
            if rng.random() < r:
                break_indices.append(i)
        break_indices.append(len(original))
        ret = list()
        for start_idx, end_idx in zip(break_indices[:-1], break_indices[1:]):
            ret.append(original[start_idx:end_idx])
        return ret
    return breaker

def get_permuter(rng: Generator) -> Callable[[List[Any]], List[Any]]:
    """Returns a function that returns a random permutation of a list."""
    def permuter(s):
        permutation = rng.permutation(len(s))
        return [s[i] for i in permutation]
    return permuter

def concatenator(x: List[List[Any]]) -> List[Any]:
    """Concatenates lists."""
    return reduce(operator.add, x)

def composed(*functions) -> Callable:
    """Returns the composition of the given functions."""
    def composition(x) -> Any:
        for f in functions:
            x = f(x)
        return x
    return composition
