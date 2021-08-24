"""Traverse the Internet via hyperlinks."""

from typing import Callable, Iterable, Optional, Set, NewType
from gcld3 import NNetLanguageIdentifier
from bs4 import BeautifulSoup
import requests
from numpy.random import Generator

UrlResolver = Callable[[str], str]

def web_walk(
    start: str,
    desired_text_len: int,
    rng: Generator,
    language: Optional[str] = None,
    websites: Optional[Set[str]] = None,
    fringe_size: int = 5,
    url_resolver: UrlResolver = lambda s: s,
    verbose: bool = False
) -> str:
    """Returns a string of length approximately `desired_text_len`
    containing text from `start` and websites linked by `start`.

    :param start: The URL of a website
    :param desired_text_len: The desired amount of text (in characters)
    :param language: The BCP-47 language code of the desired language of
        the text (see https://github.com/google/cld3 for details)
    :param websites: The set of acceptable websites to explore (e.g.,
        {https://ar.wikipedia.org})
    :param fringe_size: The desired number of websites to explore
        simultaneously
    :param url_resolver: A function for interpreting hyperlinks that
        might otherwise be invalid
    """
    visited = set()
    result = ''
    def web_walk_recursive(fringe: Iterable[str]):
        nonlocal result
        if len(result) >= desired_text_len: return
        new_fringe = list()
        for url in fringe:
            if verbose: print('Visiting {}...'.format(url))
            visited.add(url)
            try:
                response = requests.get(url_resolver(url))
            except:
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            text =  soup.get_text()
            if language is None or (
                    default_cld3.FindLanguage(text).language == language):
                result += text
            new_fringe.extend([
                url_resolver(a.get('href'))
                for a in soup.find_all('a')
                if a.get('href') is not None and a.get('href') not in visited
            ])
        if websites is not None:
            new_fringe = [a for a in new_fringe if any(
                a.startswith(website + '/')
                for website in websites
            )]
        if len(new_fringe) > fringe_size:
            new_fringe = rng.choice(new_fringe, fringe_size)
        web_walk_recursive(new_fringe)
    web_walk_recursive({start})
    return result

def getPrefixer(prefix: str) -> UrlResolver:
    """Returns a `UrlResolver` that prefixes otherwise invalid URLs with
    `prefix`.
    """
    def prefixer(url: str) -> str:
        if url.startswith('http'):
            return url
        return prefix + url
    return prefixer

default_cld3 = NNetLanguageIdentifier(1, 700)