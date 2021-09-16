from typing import Any, Generic, Hashable, Iterable, Optional, Sequence, \
    TypeVar
import pytesseract
from pytesseract import TesseractError
from gcld3 import NNetLanguageIdentifier
from PIL import Image
import languages
import pycountry
import os
import fitz
import pandas as pd
from io import StringIO
import re
import csv
import time
import pickle
import warnings

'''This module handles our interface with Tesseract. It must be
remembered that Tesseract does all of the magic behind the
scenes; the goal here is simply to provide a clean interface
with Tesseract that is optimized for our use case (multiple
languages, use of IPA characters, issues with orientation,
emphasis on speed, et cetera).
'''

Item = TypeVar('Item', bound=Hashable)


class WeightTracker(Generic[Item]):
    """Tracks the weights of items and supports iteration over them in
    order.
    All objects in the universe are in any WeightTracker instance,
    regardless of whether they have been explicitly added; by default,
    their weights are 0.
    """

    def __init__(
        self,
        items: Sequence[Item],
        presorted: bool = True,
        r: float = 0.5
    ):
        """Initializes a WeightTracker that tracks the weights of ITEMS.
        :param items: the items whose weights are to be tracked
        :param presorted: whether `items` is presorted in order of
            DECREASING expected importance
        :param r: the proportion by which all weights should change in
            response to each weight update. Set to a large value (close
            to 1) to make the WeightTracker weight recent observations
            and old observations equally heavily. Set to a small value
            (close to 0) to make old observations relatively
            unimportant.
        """
        self.items = list(items)
        self.r = r
        self.weights = {
            item: (1 / (i + 1) if presorted else 0)
            for i, item in enumerate(items)
        }

    def add_weight(self, item: Item):
        """Increases the weight given to ITEM and re-orders the items by
        weight.
        """
        self.weights = {item: self.weights[item]
                        * self.r for item in self.items}
        self.weights[item] = self.weights.get(item, 0) + 1
        self.items.sort(key=lambda item: self.weights[item], reverse=True)


class Text:
    """Describes a single text that includes a coherent set of characteristics,
    such as language used.
    """
    global_possible_languages = [
        'eng', 'tur', 'ara', 'deu', 'fra', 'rus', 'spa', 'nld',
        'jpn', 'chi_sim', 'chi_tra', 'heb', 'ita', 'dan', 'swe',
        'ell', 'lat', 'fin'
    ]
    languages_by_script = {
        'Latin': {
            'eng', 'tur', 'deu', 'fra', 'spa', 'nld', 'ita', 'dan', 'swe',
            'fin'
        },
        'Arabic': {'ara'},
        'Cyrillic': {'rus'},
        # FIXME: Tesseract recognizes Greek as Cyrillic
        'Greek': {'ell'},
        'Japanese': {'jpn'},
        'Japanese_vert': {'jpn'},
        'Han': {'chi_sim', 'chi_tra'},
        'Hebrew': {'heb'},
    }
    iso2tess = {
        'en': 'eng',
        'tr': 'tur',
        'ar': 'ara',
        'de': 'deu',
        'fr': 'fra',
        'ru': 'rus',
        'es': 'spa',
        'nl': 'nld',
        'ja': 'jpn',
        'zh': 'chi_sim',
        'zh': 'chi_tra',
        'he': 'heb',
        'it': 'ita',
        'da': 'dan',
        'sv': 'swe',
        'el': 'ell',
        'la': 'lat',
        'fi': 'fin'
    }
    # This is a hack: The justification is empirical, not theoretical. I have
    # found that there exist at least some documents that can be processed with
    # much greater accuracy if their images are scaled by a factor of 2. The
    # is that the program runs significantly slower: It seems to be slowed by
    # a constant factor of perhaps 2 or 3.
    default_image_scale = 1.75
    alternate_image_scales = (2, 4)
    word_height_range = (14, 17)
    target_word_height = 15.5
    target_mean_conf = 90
    max_unreadable = 5

    def __init__(
        self,
        src: os.PathLike,
        out: os.PathLike,
        coarse_thresh=75,
        min_relative_conf=0,
        image_area_thresh=0.5,
        text_len_thresh=100,
        languages: Optional[WeightTracker] = None,
        second_languages=None,
        verbose=False
    ):
        """Initializes a Text object from the file specified at `src`.
        :param path: must lead to a directory that does not yet exist.
        :param src: the path to the file to be analyzed. When accessing
            files in Google Drive, it is recommended to access them by
            ID to circumvent name conflicts, save a temporary file in a
            different location, and pass the location of the temporary
            file as `src`.
        :param out: the path the the working directory of this Text
            object (where output is to be saved)
        :param coarse_thresh: the minimum mean confidence level by word
            required to assume that the OCR output is at least using the
            correct script and orientation
        :param relative_conf: the minimum confidence level for a
            particular word relative to the mean confidence level of the
            entire page
        :param image_area_thresh: the threshold proportion of page area
            that is consumed by an image, beyond which the image must be
            taken into acccount in the text extraction process
        :param image_area_thresh: the threshold length of the text (in
            characters) originally explicitly encoded in a PDF, beyond
            which the original text may be taken into account in the
            text extraction process
        :param languages: a WeightTracker instance with the expected
            languages weighted in accordance to their expected
            probabilities
        :param second_languages: a WeightTracker instance with the
            languages expected to appear in isolated foreign words
            (e.g., proper nouns)
        :param verbose: whether detailed information should be printed
            by this Text instance
        """
        self.src = src
        self.out = out
        self.coarse_thresh = coarse_thresh
        self.min_relative_conf = min_relative_conf
        self.image_area_thresh = image_area_thresh
        self.text_len_thresh = text_len_thresh
        self.languages = (
            WeightTracker(Text.global_possible_languages, presorted=True)
            if languages is None else languages
        )
        self.second_languages = (
            WeightTracker(Text.global_possible_languages)
            if second_languages is None else second_languages
        )
        self.verbose = verbose
        self.texts = list()
        self.metadata = list()
        self.orientations = list()
        self.page_languages = list()
        self.mean_confidences = list()
        self.used_original_texts = list()
        self.times = list()
        self.scales = list()

    def save_ocr(self):
        """Saves the OCR output to a CSV in the top level of the working
        directory of this Text object."""
        t0 = time.time()
        document = fitz.open(self.src)
        for i, page in enumerate(document):
            if self.verbose:
                print('{} out of {} pages analyzed in {:.2f} seconds...'
                      ''.format(i, len(document), time.time() - t0))
            self._analyze_page(page)
        os.mkdir(self.out)
        pd.DataFrame(data={
            'text': self.texts,
            'orientation': self.orientations,
            'language': self.page_languages,
            'mean_confidence': self.mean_confidences,
            'used_original_text': self.used_original_texts,
            'time': self.times,
            'scale': self.scales,
        }).to_csv(os.path.join(self.out, 'page.csv'))
        self.save()

    def _analyze_page(self, page: fitz.Page):
        """Analyzes `page` and records the data extracted from it. Does
        nothing if the page cannot be analyzed successfully.
        """
        original_text = page.get_text()
        if (
            total_image_area(page) / page.bound().getArea()
            < self.image_area_thresh
            and not len([a for a in original_text if a == '�'])
            > self.max_unreadable
        ):
            metadata, orientation_used, scale = None, None, None
            language = detected_language(original_text)
            self.texts.append(original_text)
            self.mean_confidences.append(None)
            used_original_text = True
        else:
            metadata, orientation_used, language, scale = self._run_ocr(
                page,
                (detected_language(original_text)
                 if len(original_text) >= self.text_len_thresh
                 else self.languages.items[0])
            )
            if mean_conf(metadata) < self.coarse_thresh:
                warnings.warn('Failed to analyze image.')
            self.texts.append(data_to_string(
                metadata.corrected if 'corrected' in metadata.columns
                else metadata.text
            ))
            self.mean_confidences.append(mean_conf(metadata))
            used_original_text = False
        self.languages.add_weight(language)
        self.metadata.append(metadata)
        self.orientations.append(orientation_used)
        self.page_languages.append(language)
        self.used_original_texts.append(used_original_text)
        self.times.append(time.time())
        self.scales.append(scale)

    def _run_ocr(
        self, page: fitz.Page, language_guess: str
    ) -> tuple[
        Optional[pd.DataFrame],
        Optional[float],
        Optional[str],
        Optional[float]
    ]:
        """Returns metadata, orientation, detected language, and image
        scale used from the analysis of `page`. Returns
        `(None, None, None, None)` upon failure to extract text from `page`.
        :param page: the page to be analyzed
        :param language_guess: - the expected language of any text in
            `image`
        """
        orientation_used = 0
        scale_used = self.default_image_scale
        image = image_from_page(page, scale=scale_used).rotate(
            orientation_used, expand=True)
        # What follows is the first pass, assuming that the page is "typical"
        try:
            metadata = data(image, language_guess)
        except TesseractError as e:
            # FIXME: Look into when this happens and if it is avoidable.
            warnings.warn('Tesseract failed: ' + str(e))
            return (None, None, None, None)
        # What follows is an OSD-assisted attempt to improve upon the first
        # pass
        if mean_conf(metadata) < self.coarse_thresh:
            if self.verbose:
                print('First guess at orientation + language failed.')
            for scale in self.alternate_image_scales:
                image = image_from_page(page, scale=scale)
                try:
                    result = self._osd_assisted_analysis(image)
                    if mean_conf(result[-1]) > mean_conf(metadata):
                        orientation_used, language_guess, metadata = result
                        scale_used = scale
                    if mean_conf(metadata) >= self.coarse_thresh:
                        break
                except (TesseractError, ManagerError) as e:
                    warnings.warn('OCR failed: ' + str(e))
        # What follows is a final pass with optimal text size and language
        # TODO: Factor all of the following out.
        median_height = metadata.height.median()
        language = detected_language(data_to_string(metadata.text))
        if language != language_guess or (
            mean_conf(metadata) < self.target_mean_conf
            and not (
                self.word_height_range[0] <= median_height <=
                self.word_height_range[1]
            )
        ):
            optimal_scale = (
                scale_used * self.target_word_height / median_height
            )
            if self.verbose:
                print('Retrying. Language={}, scale={:.4f}'.format(
                    language, optimal_scale))
            result = data(
                image_from_page(page, scale=optimal_scale).rotate(
                    orientation_used, expand=True
                ),
                language
            )
            if mean_conf(result) > mean_conf(metadata):
                metadata = result
                scale_used = optimal_scale
        return (metadata, orientation_used, language, scale_used)

    def _osd_assisted_analysis(
        self,
        image: Image,
        max_n_languages: int = 4
    ) -> tuple[float, str, pd.DataFrame]:
        """Returns the orientation, language, and metadata produced from
        analyzing `image` with orientation and script detection. Throws
        `TesseractError` or `ManagerError` upon failure.
        """
        osd_result = osd(image)
        image = image.rotate(osd_result['Orientation in degrees'], expand=True)
        if osd_result['Script'] not in Text.languages_by_script:
            raise ManagerError('The script detected by OSD, "{}", is not '
                               'supported.'.format(osd_result['Script']))
        poss_languages = Text.languages_by_script[osd_result['Script']]
        for language in self.languages.items:
            if language in poss_languages:
                return (osd_result['Orientation in degrees'], language,
                        data(image, language))

    def _correct(self, image: Image, metadata: pd.DataFrame, min_conf: float):
        """Adds a column to the metadata table `metadata` that is the
        corrected form of the words given in its "text" column.
        :param image: the image to analyze
        :param metadata: the metadata table that is to be corrected
        :param min_conf: the minimum confidence level required for a
            word to be assumed correct and excluded from further
            examination
        """
        def corrector(row):
            """Uses data in ROW corresponding to a word shown in IMAGE
            to determine the text that most likely represents the word.
            Updates weights in SECOND_LANGUAGES depending on which
            languages successfully give high-certainty matches.
            """
            if 0 <= row.conf < min_conf:
                word_image = image.crop(
                    (row.left, row.top, row.left+row.width, row.top+row.height)
                )
                for language in self.second_languages.items:
                    metadata = data(
                        word_image, language,
                        config='--psm 8'  # Expect a single word.
                    )
                    if mean_conf(metadata) >= min_conf:
                        self.second_languages.add_weight(language)
                        correct_word = data_to_string(metadata.text).strip()
                        if self.verbose:
                            print('Correcting "{}" to "{}" (lang={})'.format(
                                row.text, correct_word, language))
                        return correct_word
            else:
                return row.text
        metadata['corrected'] = metadata.apply(corrector, axis=1)

    def save(self):
        with open(os.path.join(self.out, 'analysis.pickle'), 'wb') as dbfile:
            pickle.dump(self, dbfile)


class ManagerError(Exception):
    pass


def detected_language(
    text: str,
    default: str = 'eng',
    nnli: NNetLanguageIdentifier = NNetLanguageIdentifier(1, 700)
):
    """Returns the detected language of `text`, using the LangCode
    recognized by Tesseract (as described here:
    https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html
    ).
    :param text: the text to analyze
    :param default: the language to return if no likely language can be
        found
    """
    if not text.strip():
        return default
    result = nnli.FindLanguage(text)
    if not result.probability:
        return default
    return languages.langcodes.iso_639_3_to_tess(
        pycountry.get(alpha_2=result.language).alpha_3
    )


def image_from_page(page: fitz.Page, scale: float = 1) -> Image:
    """Converts a page to an image.
    :param page: the page to be represented as an image
    :param scale: the proportion by which to scale the image
    """
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    return Image.frombytes(
        ("RGBA" if pix.alpha else "RGB"),
        [pix.width, pix.height], pix.samples
    )


def total_image_area(page: fitz.Page) -> int:
    """Returns the total area (in pixels) consumed by images that appear
    in `page`.
    Does not account for overlap between images, so it is possible for
    the total computed area to exceed the actual area of the page.
    """
    return sum(
        rect.getArea()
        for image in page.get_images()
        for rect in page.get_image_rects(image)
    )


def mean_conf(metadata: pd.DataFrame) -> float:
    """Returns the mean confidence by word of the OCR output given by
    `metadata`.
    Returns 0 if `metadata` is `None` or has nothing but whitespace.
    :param metadata: a `DataFrame` with the format of Tesseract output
    """
    if metadata is None:
        return 0
    valid_confs = metadata.conf[(metadata.conf >= 0) & pd.array([
        (not pd.isna(text) and (text.strip() != '')) for text in metadata.text
    ])]
    return valid_confs.mean() if len(valid_confs.index) > 0 else 0


def osd(image: Image) -> dict:
    """Returns orientation and script data for `image`.
    """
    s = pytesseract.image_to_osd(image)
    ret = dict()
    for line in s.split('\n'):
        if line:
            key, value = line.split(':')
            key, value = key.strip(), value.strip()
            ret[key] = appropriate_type(value)
    return ret


def appropriate_type(value: Any) -> Any:
    """Returns a representation of `value` cast to the simplest possible
    type given its content.
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def data(image: Image, language: str, config: str = '') -> pd.DataFrame:
    """Returns a `DataFrame` with the OCR output corresponding to
    `image`.
    """
    s = pytesseract.image_to_data(image, lang=language, config=config)
    return pd.read_csv(StringIO(s), sep='\t', quoting=csv.QUOTE_NONE)


def data_to_string(words: Iterable[str]):
    """Extracts a string from the metadata table column `words` that is
    identical to the one generated by `pytesseract.image_to_string`.
    Used to avoid redundant computations.
    """
    # TODO: I was suprised to find that it was necessary to cast the words
    # (direct Tesseract output) as strings. Perhaps look into this.
    text = ' '.join('\n' if pd.isna(word) else str(word) for word in words)
    single_newline = re.compile(r' \n ')
    multiple_newline = re.compile(r'( \n){2,} ')
    text = multiple_newline.sub('\n\n', text)
    text = single_newline.sub('\n', text)
    return text
