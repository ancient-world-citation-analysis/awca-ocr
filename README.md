# awca-ocr
OCR work for the Ancient World Citation Analysis project at UC Berkeley.

The main product of the work in this repository is the `Text` class. Basic usage:

```
from tesseract_manager import Text

src = 'path/to/a/pdf_file.pdf'
out = 'path/to/directory/that/will/be/created/to/store/ocr/output'

Text(src, out).save_ocr()
```
