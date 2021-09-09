# awca-ocr
OCR work for the Ancient World Citation Analysis project at UC Berkeley.

The main product of the work in this repository is the `Text` class. Basic usage:

```
from tesseract_manager import Text

src = 'path/to/a/pdf_file.pdf'
out = 'path/to/directory/that/will/be/created/to/store/ocr/output'

Text(src, out).save_ocr()
```

# Developer Notes
Here are some tips:
* If you must use Windows instead of a Unix-like system, you may wish to use a platform such as WSL or VMware Workstation (or even Google Drive and Colab) that will enable you to emulate that environment. This is because
  * the package `gcld3` is not easy to install on Windows due to its dependency on protobufs, and
  * Tesseract 4 is not easy to install on Windows.
* A [virtual environment](https://docs.python.org/3/tutorial/venv.html) might be useful.
* Changing into this git repository and running `pip install -r requirements.txt` should install all required packages.

Here are some gentle suggestions in order of decreasing importance.
* Docstrings explaining the purpose of a module, class, or function are desirable.
* There is a hope of becoming compliant with [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines. Pycodestyle and autopep8 can help: They are easy to install in VS Code, and pycodestyle is enabled by default in PyCharm.
* There is a hope of becoming compliant with reST format for docstrings.
