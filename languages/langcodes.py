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
* BCP 47 uses ISO 639-1 when possible, and can be assumed to be the same
as ISO 639-1 for the purpose of using CLD3.

The above facts, combined with this module, should be sufficient for
interpreting language codes in this project.
"""

"""The language codes used by Tesseract and their corresponding
ISO 639-2 language codes.

The two are almost identical, but there are exceptions.
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
    'chi_sim': 'chi',
    'chi_tra': 'chi',
    'chr': 'chr',
    'cos': 'cos',
    'cym': 'cym',
    'dan': 'dan',
    'dan_frak': 'dan',
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
    'kur': 'kur',
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
    'slk_frak': 'slk',
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
    'tgl': 'tgl',
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

SCRIPTS = {
    'Arabic': {'ara', 'fas', 'kur', 'pus', 'snd', 'uig', 'urd'},
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
    'Fraktur': {'dan_frak', 'frk', 'slk_frak'},  # Variant of Latin
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
    'Lao': {'lao'},
    'Latin': {
        'afr', 'aze', 'bos', 'bre', 'cat', 'ceb', 'ces', 'cos', 'cym', 'dan',
        'eng', 'enm', 'epo', 'est', 'eus', 'fao', 'fil', 'fin', 'fra', 'frm',
        'fry', 'gla', 'gle', 'glg', 'hat', 'hrv', 'hun', 'iku', 'ind', 'isl',
        'ita', 'ita_old', 'jav', 'kmr', 'lat', 'lav', 'lit', 'ltz', 'mlt',
        'mri', 'msa', 'nld', 'nor', 'oci', 'pol', 'por', 'que', 'ron', 'slk',
        'slv', 'spa', 'spa_old', 'sun', 'sqi', 'srp_latn', 'swa', 'swe', 'tgl',
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


def inverse(d: dict) -> dict[set]:
    """Computes the inverse of a (possibly not injective) map."""
    ret = dict()
    for key, value in d.items():
        to_add = key if isinstance(key, set) else {key}
        ret[value] = ret.get(value, set()) | to_add
    return ret


def iso_639_3_to_tess(langcode: str) -> set[str]:
    """Converts `langcode`(s) to a language code that is recognized by
    Tesseract.
    """
    tesseract_inverse = getattr(iso_639_3_to_tess, 'tesseract_inverse', None)
    if not tesseract_inverse:
        tesseract_inverse = inverse(TESSERACT)
        iso_639_3_to_tess.tesseract_inverse = tesseract_inverse
    return tesseract_inverse[langcode]
