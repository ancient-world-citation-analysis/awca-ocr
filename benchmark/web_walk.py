"""Traverse the Internet via hyperlinks."""

from typing import Callable, Iterable, Optional, Set
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
        exclude=(
          'style', 'script', 'img', 'meta', 'nav', 'figure', 'figcaption',
          'figure', 'code', 'data', 'var', 'audio', 'video', 'map', 'video',
          'iframe', 'embed', 'object', 'param', 'picture', 'portal', 'math',
          'svg', 'canvas', 'table', 'base', 'head', 'link', 'kbd', 'area',
          'track', 'source', 'slot', 'template', 'form', 'details', 'dialog',
          'menu', 'summary'
        ),
        verbose: bool = False
) -> str:
    """Returns a string of length approximately `desired_text_len`
    containing text from `start` and websites linked by `start`.

    :param start: The URL of a website
    :param desired_text_len: The desired amount of text (in characters)
    :param rng: The random number generator that determines which sites
        are visited
    :param language: The BCP-47 language code of the desired language
        of the text (see https://github.com/google/cld3 for details)
    :param websites: The set of acceptable websites to explore (e.g.,
        {https://ar.wikipedia.org})
    :param fringe_size: The desired number of websites to explore
        simultaneously
    :param url_resolver: A function for interpreting hyperlinks that
        might otherwise be invalid
    :param exclude: HTML subtree types to exclude from the output, as
        denoted by their HTML tags
    :param verbose: Whether to print verbose output
    """
    visited = set()

    def filtered(soup: BeautifulSoup) -> Iterable[str]:
        return {
            resolved for resolved in [
                url_resolver(a.get('href'))
                for a in soup.find_all('a')
                if a.get('href') is not None
            ]
            if resolved not in visited and (
                websites is None or any(
                    resolved.startswith(website + '/')
                    for website in websites
                )
            )
        }

    def checked(text: str) -> str:
        if language is None:
            return text
        if default_cld3.FindLanguage(text).language == language:
            return text
        return ''

    def web_walk_recursive(fringe: Iterable[str], result='') -> str:
        if len(result) >= desired_text_len:
            return result
        new_fringe = list()
        for url in fringe:
            if verbose:
                print('Visiting {}...'.format(url))
            visited.add(url)
            try:
                response = requests.get(url_resolver(url))
            except requests.exceptions.RequestException:
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup.find_all(list(exclude)):
                tag.extract()
            text = soup.get_text()
            result += checked(text)
            new_fringe.extend(filtered(soup))
        if len(new_fringe) > fringe_size:
            new_fringe = rng.choice(new_fringe, fringe_size)
        return web_walk_recursive(new_fringe, result)

    return web_walk_recursive({start})


def get_prefixer(
        prefix: str, resolver: UrlResolver = lambda x: x
) -> UrlResolver:
    """Returns a `UrlResolver` that prefixes otherwise invalid URLs with
    `prefix`.
    """

    def prefixer(url: str) -> str:
        url = resolver(url)
        if url.startswith('http'):
            return url
        return prefix + url

    return prefixer


def get_query_string_remover(
    resolver: UrlResolver = lambda x: x
) -> UrlResolver:
    """Returns a `UrlResolver` that removes the query strings from
    URLs.
    """
    return lambda x: resolver(x).split('?')[0]


def wikipedia_about_page(langcode: str) -> str:
    """Returns the Wikipedia "About" page for the language given by
    `langcode`.
    :param langcode: The language code of the desired language
    """
    return 'https://{}.wikipedia.org/wiki/Wikipedia:About'.format(langcode)


def wikipedia(langcode: str) -> str:
    """Returns the Wikipedia website for the language given by
    `langcode`.
    :param langcode: The language code of the desired language
    """
    return 'https://{}.wikipedia.org'.format(langcode)


default_cld3 = NNetLanguageIdentifier(1, 700)
