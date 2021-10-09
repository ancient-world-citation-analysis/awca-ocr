"""Language codes are provided here, in the forms stipulated by specific
tools.

For more generally applicable manipulation of language codes,
`pycountry` is recommended. Note that
* "alpha_2" refers to the same standard as "ISO 639-1"
* "alpha_3" refers to the same standard as "ISO 639-3"
* ISO 639-3 includes all of the same _languages_ as ISO 639-2/T, with
the same language codes, although it is not formally a superset because
ISO 639-2 includes non-languages. See
https://en.wikipedia.org/wiki/ISO_639-3#Collective_languages for
details.
* BCP 47 uses ISO 639-1 when possible, but less concise standards when
necessary.

The above facts, combined with this module, should be sufficient for
interpreting language codes in this project.
"""

from typing import Any, Dict, Set
import pycountry


"""A comprehensive list of the languages supported by Tesseract 4
and their corresponding ISO 639-2/T language codes.
"""
TESSERACT = {
    'afr': 'afr',
    'amh': 'amh',
    'ara': 'ara',
    'asm': 'asm',
    'aze': 'aze',
    'aze_cyrl': 'aze',
    'bel': 'bel',
    'ben': 'ben',
    'bod': 'bod',
    'bos': 'bos',
    'bre': 'bre',
    'bul': 'bul',
    'cat': 'cat',
    'ceb': 'ceb',
    'ces': 'ces',
    'chi_sim': 'zho', # One of the few instances where ISO 639-2/B
    'chi_tra': 'zho', # (legacy) is chosen over ISO 639-2/T
    'chr': 'chr',
    'cos': 'cos',
    'cym': 'cym',
    'dan': 'dan',
    # 'dan_frak': 'dan',  # Not available in Tesseract 4
    'deu': 'deu',
    # 'deu_frak': 'deu',  # Not available in Tesseract 4
    'dzo': 'dzo',
    'ell': 'ell',
    'eng': 'eng',
    'enm': 'enm',
    'epo': 'epo',
    'est': 'est',
    'eus': 'eus',
    'fao': 'fao',
    'fas': 'fas',
    'fil': 'fil',
    'fin': 'fin',
    'fra': 'fra',
    'frk': 'frk',
    'frm': 'frm',
    'fry': 'fry',
    'gla': 'gla',
    'gle': 'gle',
    'glg': 'glg',
    'grc': 'grc',
    'guj': 'guj',
    'hat': 'hat',
    'heb': 'heb',
    'hin': 'hin',
    'hrv': 'hrv',
    'hun': 'hun',
    'hye': 'hye',
    'iku': 'iku',
    'ind': 'ind',
    'isl': 'isl',
    'ita': 'ita',
    'ita_old': 'ita',
    'jav': 'jav',
    'jpn': 'jpn',
    'kan': 'kan',
    'kat': 'kat',
    'kat_old': 'kat',
    'kaz': 'kaz',
    'khm': 'khm',
    'kir': 'kir',
    'kmr': 'kmr',
    'kor': 'kor',
    'kor_vert': 'kor',
    'kur_ara': 'kur',  # This came as a surprise: the language code
                       # given in the documentation is "kur"
    'lao': 'lao',
    'lat': 'lat',
    'lav': 'lav',
    'lit': 'lit',
    'ltz': 'ltz',
    'mal': 'mal',
    'mar': 'mar',
    'mkd': 'mkd',
    'mlt': 'mlt',
    'mon': 'mon',
    'mri': 'mri',
    'msa': 'msa',
    'mya': 'mya',
    'nep': 'nep',
    'nld': 'nld',
    'nor': 'nor',
    'oci': 'oci',
    'ori': 'ori',
    'pan': 'pan',
    'pol': 'pol',
    'por': 'por',
    'pus': 'pus',
    'que': 'que',
    'ron': 'ron',
    'rus': 'rus',
    'san': 'san',
    'sin': 'sin',
    'slk': 'slk',
    # 'slk_frak': 'slk',  # Not available in Tesseract 4
    'slv': 'slv',
    'snd': 'snd',
    'spa': 'spa',
    'spa_old': 'spa',
    'sqi': 'sqi',
    'srp': 'srp',
    'srp_latn': 'srp',
    'sun': 'sun',
    'swa': 'swa',
    'swe': 'swe',
    'syr': 'syr',
    'tam': 'tam',
    'tat': 'tat',
    'tel': 'tel',
    'tgk': 'tgk',
    # 'tgl': 'tgl',  # Not available in Tesseract 4
    'tha': 'tha',
    'tir': 'tir',
    'ton': 'ton',
    'tur': 'tur',
    'uig': 'uig',
    'ukr': 'ukr',
    'urd': 'urd',
    'uzb': 'uzb',
    'uzb_cyrl': 'uzb',
    'vie': 'vie',
    'yid': 'yid',
    'yor': 'yor'
}


"""A comprehensive list of the scripts supported by Tesseract 4
and their corresponding ISO 639-2/T language codes.
"""
SCRIPTS = {
    'Arabic': {'ara', 'fas', 'kur_ara', 'pus', 'snd', 'uig', 'urd'},
    'Armenian': {'hye'},
    'Bengali': {'asm', 'ben'},
    'Canadian_Aboriginal': {},
    'Cherokee': {'chr'},
    # kaz is historically Cyrillic, but transitioning to Latin
    'Cyrillic': {
        'aze_cyrl', 'bel', 'bul', 'kaz', 'kir', 'mkd', 'mon', 'rus', 'srp',
        'tat', 'tgk', 'ukr', 'uzb_cyrl'
    },
    'Devanagari': {'hin', 'mar', 'nep', 'san'},
    'Ethiopic': {'amh', 'tir'},
    'Fraktur': {  # Variant of Latin script
        # 'dan_frak',
        # 'deu_frak',
        'frk',
        # 'slk_frak'
    },
    'Georgian': {'kat', 'kat_old'},
    'Greek': {'ell', 'grc'},
    'Gujarati': {'guj'},  # Variant of Devanagari
    'Gurmukhi': {'pan'},
    'HanS': {'chi_sim'},
    'HanS_vert': {'chi_sim'},
    'HanT': {'chi_tra'},
    'HanT_vert': {'chi_tra'},
    'Hangul': {'kor'},
    'Hangul_vert': {'kor_vert'},
    'Hebrew': {'heb', 'yid'},
    'Japanese': {'jpn'},
    'Japanese_vert': {'jpn'},
    'Kannada': {'kan'},  # Similar to Telugu & to a lesser extent Sinhala
    'Khmer': {'khm'},
    'Korean': {'kor'},  # Same as Hangul. It is strange that both exist. It is
                        # also strange that the Tesseract documentation
                        # excludes "Korean" for the list of supported scripts,
                        # even though its OSD feature reports "Korean".
    'Lao': {'lao'},
    'Latin': {
        'afr', 'aze', 'bos', 'bre', 'cat', 'ceb', 'ces', 'cos', 'cym', 'dan',
        'eng', 'enm', 'epo', 'est', 'eus', 'fao', 'fil', 'fin', 'fra', 'frm',
        'fry', 'gla', 'gle', 'glg', 'hat', 'hrv', 'hun', 'iku', 'ind', 'isl',
        'ita', 'ita_old', 'jav', 'kmr', 'lat', 'lav', 'lit', 'ltz', 'mlt',
        'mri', 'msa', 'nld', 'nor', 'oci', 'pol', 'por', 'que', 'ron', 'slk',
        'slv', 'spa', 'spa_old', 'sun', 'sqi', 'srp_latn', 'swa', 'swe',
        # 'tgl',  # Not available in Tesseract 4
        'ton', 'tur', 'uzb', 'vie', 'yor'
    },
    'Malayalam': {'mal'},
    'Myanmar': {'mya'},
    'Oriya(Odia)': {},
    'Sinhala': {'sin'},
    'Syriac': {'syr'},
    'Tamil': {'tam'},
    'Telugu': {'tel'},
    'Thaana': {},
    'Thai': {'tha'},
    'Tibetan': {'bod', 'dzo'},
    'Vietnamese': {}
}


"""A (possibly non-comprehensive) mapping from deprecated ISO codes
that may appear in BCP-47 to the shortest available non-deprecated
ISO codes (avoiding 639-2/B).
"""
DEPRECATED_TO_CURRENT = {
    'bh': 'bih',  # Bihari. There is no acceptable 2-letter code.
    'in': 'id',  # Indonesian
    'iw': 'he',  # Hebrew
    'ji': 'yi',  # Yiddish
    'jw': 'jv',  # Javanese
    'mo': 'ro',  # Moldovan
    'sh': 'sr'  # Serbo-Croatian -> Serbian. Not quite right,
                # but the closest approximation possible.
}


def memoize(f):
    """Memoizes `f`, a function with one argument.
    """
    arg = []
    out = []
    def memoized(x):
        for x_, y in zip(arg, out):
            if x is x_: # Memoize wrt identity equality
                return y
        arg.append(x)
        out.append(f(x))
        return out[-1]
    return memoized


@memoize
def inverse(d: dict) -> Dict[Any, set]:
    """Computes the inverse of a (possibly not injective) map."""
    ret = dict()
    for key, value in d.items():
        to_add = key if isinstance(key, set) else {key}
        ret[value] = ret.get(value, set()) | to_add
    return ret


def iso_639_3_to_tess(langcode: str) -> Set[str]:
    """Converts `langcode`(s) to a language code that is recognized by
    Tesseract.
    """
    tesseract_inverse = getattr(iso_639_3_to_tess, 'tesseract_inverse', None)
    if not tesseract_inverse:
        tesseract_inverse = inverse(TESSERACT)
        iso_639_3_to_tess.tesseract_inverse = tesseract_inverse
    return inverse(TESSERACT)[langcode]


def bcp47_to_tess(bcp47, default):
    """Converts the BCP-47-style language code `bcp47` to a language
    code that is recognized by Tesseract.
    Returns `default` if no such language code exists.
    """
    base_bcp47 = bcp47.split('-')[0].lower()
    base_bcp47 = DEPRECATED_TO_CURRENT.get(base_bcp47, base_bcp47)
    if len(base_bcp47) == 2:
        language = pycountry.languages.get(alpha_2=base_bcp47)
        if language is None:
            print("WARNING: No language found corresponding to " + base_bcp47)
            return default
        iso_639_3 = language.alpha_3
    else:
        iso_639_3 = base_bcp47
    # Heuristic: Often, the language with the shortest langcode
    # is the one without modifiers and therefore probably the
    # most common or general one. For instance 'ita' is more
    # common and general than 'ita_old'.
    try:
        possibilities = iso_639_3_to_tess(iso_639_3)
    except KeyError:
        return default
    return (
        min(possibilities, key=lambda language: len(language))
        if possibilities else default
    )
