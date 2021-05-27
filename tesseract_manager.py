'''This module handles our interface with Tesseract. It must be
remembered that Tesseract does all of the magic behind the
scenes; the goal here is simply to provide a clean interface
with Tesseract that is optimized for our use case (multiple
languages, use of IPA characters, issues with orientation, high
emphasis on speed, et cetera).
'''

import pytesseract
from pytesseract import TesseractError
from PIL import Image
import os
import fitz
import pandas as pd
from io import StringIO
import re
import langdetect
from langdetect.lang_detect_exception import LangDetectException
import csv
import time
import pickle
import warnings

class WeightTracker:
  """Tracks the weights of items and supports iteration over them in
  order.
  All objects in the universe are in any WeightTracker instance,
  regardless of whether they have been explicitly added; by default, their
  weights are 0.
  """
  def __init__(self, items, presorted=True, r=0.5):
    """Initializes a WeightTracker that tracks the weights of ITEMS.
    ITEMS - the items whose weights are to be tracked. These items must be
            hashable.
    PRESORTED - whether ITEMS is presorted in order of DECREASING expected
            importance
    R     - the proportion by which all weights should in response to
            each weight update. Set to a large value (close to 1) to make the
            WeightTracker weight recent observations and old observations
            equally heavily. Set to a small value (close to 0) to make old
            observations relatively unimportant.
    """
    self.items = items
    self.r = r
    self.weights = {
        item: (1 / (i + 1) if presorted else 0)
        for i, item in enumerate(items)
    }
  def add_weight(self, item):
    """Increases the weight given to ITEM.
    """
    self.weights = {item: self.weights[item] * self.r for item in self.items}
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
      'Latin': {'eng', 'tur', 'deu', 'fra', 'spa', 'nld', 'ita', 'dan', 'swe', 'fin'},
      'Arabic': {'ara'},
      'Cyrillic': {'rus'}, # Tesseract seems to recognize Greek as
                           # Cyrillic.
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
  # much greater accuracy if their images are scaled by a factor of 2. The cost
  # is that the program runs significantly slower: It seems to be slowed by a
  # constant factor of perhaps 2 or 3.
  default_image_scale = 2
  target_word_height = (14, 17)
  def __init__(self, src, out,
               coarse_thresh=75, min_relative_conf=0,
               image_area_thresh=0.5, text_len_thresh=100,
               languages=None, second_languages=None,
               verbose=False):
    """Initializes a Text object from the file specified at SRC.
    PATH must lead to a directory that does not yet exist.
    SRC - the path to the file to be analyzed. When accessing files in Google
          Drive, it is recommended to access them by ID to circumvent name
          conflicts, save a temporary file in a different location, and pass the
          location of the temporary file as SRC.
    OUT - the path the the working directory of this Text object (where output
          is to be saved)
    COARSE_THRESH - the minimum mean confidence level by word required to assume
          that the OCR output is at least using the correct script and
          orientation
    MIN_RELATIVE_CONF - the minimum confidence level for a particular word
          relative to the mean confidence level of the entire page
    IMAGE_AREA_THRESH - the threshold proportion of page area that is
          consumed by an image, beyond which the image must be taken into
          acccount in the text extraction process
    TEXT_LEN_THRESH - the threshold length of the text (in characters)
          originally explicitly encoded in a PDF, beyond which the original text
          may be taken into account in the text extraction process
    LANGUAGES - a WeightTracker instance with the expected languages weighted
          in accordance to their expected probabilities
    SECOND_LANGUAGES - a WeightTracker instance with the languages expected to
          appear in isolated foreign words (e.g., proper nouns)
    VERBOSE - whether detailed information should be printed by this Text
          instance
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
  def save_ocr(self):
    """Saves the OCR output to a CSV in the top level of the working directory
    of this Text object."""
    t0 = time.time()
    document = fitz.open(self.src)
    for i, page in enumerate(document):
      if self.verbose:
        print('{} out of {} pages analyzed in {:.2f} seconds...'.format(
            i, len(document), time.time() - t0))
      self._analyze_page(page)
    os.mkdir(self.out)
    pd.DataFrame(data={
        'text': self.texts,
        'orientation': self.orientations,
        'language': self.page_languages,
        'mean_confidence': self.mean_confidences,
        'used_original_text': self.used_original_texts,
        'time': self.times,
    }).to_csv(os.path.join(self.out, 'page.csv'))
    self.save()
  def _save_images(self):
    """Saves images of the pages of the text in the working directory."""
    os.mkdir(self.images_dir)
    convert_from_path(self.src, output_folder=self.images_dir)
  def clean(self):
    """Deletes the files needed for intermediate steps in the analysis of the
    text.
    """
    os.rmdir(self.images_dir)
  def _analyze_page(self, page):
    """Analyzes PAGE and records the data extracted from it. Does nothing if the
    page cannot be analyzed successfully.
    """
    # Guarantee: metadata, orientation_used, language, used_original_text
    # are defined after this branch statement.
    original_text = page.get_text()
    if total_image_area(page) / page.bound().getArea() < self.image_area_thresh:
      metadata, orientation_used = None, None
      language = detected_language(original_text)
      self.texts.append(original_text)
      self.mean_confidences.append(None)
      used_original_text = True
    else:
      metadata, orientation_used, language = self._run_ocr(
          image_from_page(page, scale=self.default_image_scale),
          (detected_language(original_text)
           if len(original_text) >= self.text_len_thresh
           else self.languages.items[0])
      )
      if mean_conf(metadata) < self.coarse_thresh:
        warnings.warn('Failed to analyze image.')
      else: pass
        # WORD CORRECTION FEATURE DISABLED. TODO: fix the feature.
        # self._correct(image, metadata, mean_conf(metadata)+self.min_relative_conf)
      self.texts.append(data_to_string(
          metadata.corrected if 'corrected' in metadata.columns else metadata.text
      ))
      self.mean_confidences.append(mean_conf(metadata))
      used_original_text = False
    self.languages.add_weight(language)
    self.metadata.append(metadata)
    self.orientations.append(orientation_used)
    self.page_languages.append(language)
    self.used_original_texts.append(used_original_text)
    self.times.append(time.time())
  def _run_ocr(self, image, language_guess):
    """Returns metadata, orientation detected, and language detected from the
    analysis of IMAGE. Returns (None, None, None) upon failure to extract text
    from IMAGE.
    IMAGE - the image to be analyzed
    LANGUAGE_GUESS - the expected language of any text in IMAGE
    """
    orientation_used = 0
    try:
      metadata = data(image, language_guess)
    except TesseractError as e:
      warnings.warn('Tesseract failed: ' + str(e))
      return (None, None, None)
    if mean_conf(metadata) < self.coarse_thresh:
      if self.verbose: print('First guess at orientation + language failed.')
      try:
        result = self._osd_assisted_analysis(image)
        if mean_conf(result[-1]) > mean_conf(metadata):
            image, orientation_used, language_guess, metadata = result
      except (TesseractError, ManagerError) as e:
        warnings.warn('OCR failed: ' + str(e))
    language = detected_language(data_to_string(metadata.text))
    if language != language_guess:
      if self.verbose:
        print('Retrying with detected language. Language={}'.format(language))
      metadata = data(image, language)
    return (metadata, orientation_used, language)
  def _osd_assisted_analysis(self, image):
    """Returns the image, orientation, language, and metadata produced from
    analyzing IMAGE with orientation and script detection. Throws TesseractError
    or ManagerError upon failure.
    """
    osd_result = osd(image)
    image = image.rotate(osd_result['Orientation in degrees'])
    if osd_result['Script'] not in Text.languages_by_script:
      raise ManagerError('The script detected by OSD, "{}", is not '
          'supported.'.format(osd_result['Script']))
    poss_languages = Text.languages_by_script[osd_result['Script']]
    for language in self.languages.items:
      if language in poss_languages:
        return (image, osd_result['Orientation in degrees'], language,
            data(image, language))
  def _correct(self, image, metadata, min_conf):
    """Adds a column to the metadata table METADATA that is the corrected form
    of the words given in its "text" column.
    METADATA - the metadata table that is to be corrected
    MIN_CONF - the minimum confidence level required for a word to be assumed
              correct and excluded from further examination
    LANGUAGES - a WeightTracker instance containing the possible languages,
              encoded as Tesseract LangCodes
    """
    def corrector(row):
      """Uses data in ROW corresponding to a word shown in IMAGE to determine
      the text that most likely represents the word. Updates weights in
      SECOND_LANGUAGES depending on which languages successfully give
      high-certainty matches.
      """
      if 0 <= row.conf < min_conf:
        word_image = image.crop(
            (row.left, row.top, row.left+row.width, row.top+row.height))
        for language in self.second_languages.items:
          metadata = data(word_image, language,
              config='--psm 8') # This tells Tesseract to expect a single word.
          print('DEBUG: For language={}, confidence={} and guess={}'.format(
              language, mean_conf(metadata), data_to_string(metadata.text).strip()
          ))
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

class ManagerError(Exception): pass

def detected_language(text, default='eng'):
  """Returns the detected language of TEXT, using the LangCode recognized by
  Tesseract (as described here:
  https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html)
  TEXT - the text to analyze
  DEFAULT - the language to return if no likely language can be found
  """
  assert isinstance(text, str)
  try:
    # Output of DETECT_LANGS is in decreasing order by estimated probability
    for lang in langdetect.detect_langs(text):
      if lang.lang in Text.iso2tess:
        return Text.iso2tess[lang.lang]
  except LangDetectException:
    pass
  return default
  
def image_from_page(page, scale=1):
  """Returns a PIL Image representation of PAGE, a fitz.Page object.
  PAGE  - the page to be represented as an Image
  SCALE - the proportion by which to scale the Image
  """
  pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
  return Image.frombytes(
      ("RGBA" if pix.alpha else "RGB"),
      [pix.width, pix.height], pix.samples
      )

def total_image_area(page):
  """Returns the total area (in pixels) consumed by images that appear in PAGE,
  a fitz.Page object.
  Does not account for possible overlapping between images.
  """
  return sum(
      rect.getArea()
      for image in page.get_images()
      for rect in page.get_image_rects(image)
      )

def mean_conf(metadata):
  """Returns the mean confidence by word of the OCR output given by METADATA.
  Returns 0 of METADATA is None.
  """
  if metadata is None: return 0
  return metadata.conf[metadata.conf >= 0].mean()

def osd(image):
  """Returns a dictionary with orientation and script data for IMAGE."""
  s = pytesseract.image_to_osd(image)
  ret = dict()
  for line in s.split('\n'):
    if line:
      key, value = line.split(':')
      key, value = key.strip(), value.strip()
      ret[key] = appropriate_type(value)
  return ret
def appropriate_type(value):
  """Returns a representation of VALUE cast to the appropriate type."""
  try:
    return int(value)
  except ValueError:
    try:
      return float(value)
    except ValueError:
      return value
def data(image, language, config=''):
  """Returns a DataFrame with the OCR output corresponding to IMAGE along with
  additional metadata.
  """
  s = pytesseract.image_to_data(image, lang=language, config=config)
  return pd.read_csv(StringIO(s), sep='\t', quoting=csv.QUOTE_NONE)
def data_to_string(words):
  """Extracts a string from the metadata table column WORDS that is identical
  to the one generated by pytesseract.image_to_string. Used to avoid
  redundant computations.
  """
  # TODO: I was suprised to find that it was necessary to cast the words (direct
  # Tesseract output) as strings. Perhaps look into this.
  text = ' '.join('\n' if pd.isna(word) else str(word) for word in words)
  single_newline = re.compile(r' \n ')
  multiple_newline = re.compile(r'( \n){2,} ')
  text = multiple_newline.sub('\n\n', text)
  text = single_newline.sub('\n', text)
  return text