import pytesseract
from pytesseract import Output
from tesseract_manager import data_to_string
from gensim.utils import deaccent

universal_chars = set('`1234567890-=~!@#$%^&*()_+[]\\;\',./{}|:"<>?')

charsets = {
  'ces': set('AÁBCČDĎEÉĚFGHChIÍJKLMNŇOÓPRŘSŠTŤUÚŮVXYÝZŽaábcčdďeéěfghchiíjklmnňo'
             'óprřsštťuúůvxyýzž0123456789') | universal_chars,
  # Q and W omitted from Czech b/c they are seldom used. X could also be omitted.
  'eng': set('AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz'
            ) | universal_chars,
  'lav': set('AĀBCČDEĒFGĢHIĪJKĶLĻMNŅOPRSŠTUŪVZŽaābcčdeēfgģhiījkķlļmnņoprsštuūvz'
             'ž0123456789') | universal_chars,
  'yor': set('ABDEẸFGGbHIJKLMNOỌPRSṢTUWYabdeẹfggbhijklmnoọprsṣtuwy0123456789'
             #'ÁÀÉÈẸẸ́Ẹ̀ÍÌŃǸḾM̀ÓÒỌỌ́Ọ̀ÚÙṢáàéèẹẹ́ẹ̀íìńǹḿm̀óòọọ́ọ̀ùṣ') | universal_chars,
            ) | universal_chars,
  'vie': set(#'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴ'
             #'ẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵ'
             #'ỶỷỸỹ'
             'ÀÁÈÉÊÌÍÒÓÔÙÚÝàáâèéìíòóôùúýĂăẠạẢảẬậ'
             'ẶặẸẹẺẻỆệỈỉỊịỌọỎỏỘộỤụỦủỲỳỴỵ'
             'Ỷỷ'
            ) | universal_chars,
  # Macrons omitted for Yoruba b/c they are optional and perhaps seldom used.
  # ú has been omitted for Yoruba b/c it was not being detected. Soon after all
  # the acutes & graves were omitted for Yoruba. I can't justify
  # this except on empirical grounds. :(
}

char_lookalikes = [
  set('8S¥sš'),
  set('I[|{/\\'),
  set('I]|}/\\'),
  set('|l1'),
  set('.,'),
  set('—-'),
]

def char_kin(char0, char1):
  """Returns True iff char1 is akin to char2. (This relationship is symmetric.)
  """
  return deaccent(char0) == deaccent(char1) or any(
    (char0 in s and char1 in s) for s in char_lookalikes)

def choose_char(chars):
  """Returns the detected character that is most likely to be the true
  character.
  :param chars: a sequence of (LangCode, character) pairs
  """
  weights = {
    c: (len(chars) - 1 - i) / (len(chars) ** 2)
    for i, (_, c) in enumerate(chars)
    if c != ''
  }
  for language, c in chars:
    if language in charsets:
      weights[c] += 1 / len(chars) # This is just a tie-breaker.
      for other in weights:
        if other in charsets[language] and other != c:
          weights[other] -= 1
  ret = max(weights.keys(), key=lambda c: weights[c])
  print('DEBUG: chars={}, choosing {}'.format(str(chars), ret))
  return ret
  
def box_sim(left0, bottom0, right0, top0, left1, bottom1, right1, top1):
  """Returns the Jaccard similarity of box 0 and box 1.

  >>> round(box_sim(1, 1, 3, 4, 2, 2, 5, 6), 3)
  0.125
  """
  def area(left, bottom, right, top):
    return (right - left) * (top - bottom)
  left2, bottom2 = max(left0, left1), max(bottom0, bottom1)
  right2, top2 = min(right0, right1), min(top0, top1)
  if left2 >= right2 or bottom2 >= top2:
    return 0
  intersection = area(left2, bottom2, right2, top2)
  return intersection / (
      area(left0, bottom0, right0, top0) + area(left1, bottom1, right1, top1)
      - intersection)

def translit(image, box_sim_thresh=0.65):
  """Returns the string representation of the transliteration found in IMAGE.
  :param image: an image of a transliterated Semitic language
  :param box_sim_thresh: the minimum Jaccard similarity of the bounding boxes of
      two characters for them to be recognized as describing the same character
  """
  # Variable declarations follow.
  # The following is a parameter of a rather queer sort
  flagship_language = 'ces'
  other_languages = [k for k in charsets.keys() if k != flagship_language]
  data = pytesseract.image_to_data(
    image, lang=flagship_language, output_type=Output.DICT)
  flagship_boxes = pytesseract.image_to_boxes(
    image, lang=flagship_language, output_type=Output.DICT)
  other_boxes = {
    language: pytesseract.image_to_boxes(
      image, lang=language, output_type=Output.DICT)
    for language in other_languages
  }
  # Discovery of the correct characters follows.
  chars = list()
  for char0, box0 in zip(flagship_boxes['char'], zip(
      flagship_boxes['left'], flagship_boxes['bottom'],
      flagship_boxes['right'], flagship_boxes['top']
      )):
    matches = [(flagship_language, char0)]
    for other in other_languages:
      for char1, box1 in zip(other_boxes[other]['char'], zip(
          other_boxes[other]['left'], other_boxes[other]['bottom'],
          other_boxes[other]['right'], other_boxes[other]['top']
          )):
        sim = box_sim(*box0, *box1)
        if char1 in charsets[other] and (
            sim >= box_sim_thresh or (sim > 0 and char_kin(char0, char1))):
          matches.append((other, char1))
    chars.append((char0, box0, choose_char(matches)))
  # Matching of characters to words follows.
  for i, (left, top, width, height, conf, word0) in enumerate(zip(
      data['left'], data['top'], data['width'], data['height'],
      data['conf'], data['text']
      )):
    if int(conf) >= 0:
      right, bottom = left + width, top + height
      top, bottom = image.height - top, image.height - bottom
      word = ''
      for char0, box, char in chars:
        if (char0 in word0) and box_sim(*box, left, bottom, right, top) > 0:
          word += char
      data['text'][i] = word
  return data_to_string(data['text'])











