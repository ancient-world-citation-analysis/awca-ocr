# awca-ocr
OCR work for the Ancient World Citation Analysis project at UC Berkeley.

The main product of the work in this repository is the `Text` class. Basic usage:

```
from tesseract_manager import Text

src = 'path/to/a/pdf_file.pdf'
out = 'path/to/directory/that/will/be/created/to/store/ocr/output'

Text(src, out).save_ocr()
```

The output directory will then contain two files:
* A CSV file containing text and page-level data extracted from the PDF
* A pickle containing the serialized `Text` instance that you created. This object includes the same information as the CSV, as well as word-level metadata.

# Dependencies
* `tesseract-ocr-all`. On Linux and Google Colab, this can be installed using `apt-get install tesseract-ocr-all`.
* Protobufs. On Linux and Google Colab, this can be installed using `apt-get install protobuf-compiler libprotobuf-dev`
* Various Python packages. These can be installed by changing into this Git repository and running `pip install -r requirements.txt`.
  * In Google Colab, `pip install -r reduced-requirements.txt` should be used instead. This is because many of the dependencies are pre-installed in Colab, and it is best not to install them again.

# Developer Notes
Here are some resources:
* Certain files are useful but too large or cumbersome to check into version control. These include:
  * A [1600-page PDF](https://drive.google.com/file/d/1gN47TR_KSJxMjGIp1UGOp5qavxGjFTre/view?usp=sharing) including random pages from our collection. (These are pages selected uniformly at random from PDFs selected uniformly at random.)
  * Three 400-page development sets: [sample 0](https://drive.google.com/file/d/19OQOaOqkOJTCBwS7829HHMqscgMxqUKH/view?usp=sharing), [sample 1](https://drive.google.com/file/d/1KCTdlQWOB2Nyn06EW7UW7BTZhHDYJhGm/view?usp=sharing), and [sample 2](https://drive.google.com/file/d/1bWlORa_P_dCKZhgmjumQ_uhh_C2uo_Xy/view?usp=sharing).
  * [A spreadsheet with data about the 1600-page PDF](https://docs.google.com/spreadsheets/d/1w-eGo7Rep4QbhUl5yfNXdrHs_yOqRhpn7XmwICC9CXg/edit?usp=sharing), including which pages are from PDFs that are also represented in the development sets. This spreadsheet is a source of statistics about our corpus that may guide decisionmaking for OCR.
  * [Training data for tesseract](https://drive.google.com/drive/folders/1-URSGgpsxAv5F6TQiczHXxOIMNkzr4ea?usp=sharing), and other Tesseract-related files.

Here are some tips:
* If you must use Windows instead of a Unix-like system, you may wish to use a platform such as WSL or VMware Workstation (or even Google Drive and Colab) that will enable you to emulate that environment. This is because
  * the package `gcld3` is not easy to install on Windows due to its dependency on protobufs, and
  * Tesseract 4 is not easy to install on Windows.
* A [virtual environment](https://docs.python.org/3/tutorial/venv.html) might be useful.
* Because Google Drive is being used to store datasets that are too large for a personal computer, it can be useful to access this repository via Google Colab. Because Colab notebooks essentially provide command-line access to a virtual machine (just precede commands with the symbols ! or %), it is convenient to use Git on Google Colab.

Here are some gentle suggestions in order of decreasing importance.
* Docstrings explaining the purpose of a module, class, or function are desirable.
* Compliance with [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines is desirable. Pycodestyle and autopep8 can help: They are easy to install in VS Code, and pycodestyle is enabled by default in PyCharm.
* Compliance with reST format for docstrings is desirable.
