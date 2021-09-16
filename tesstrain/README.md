Steps:
* Follow the steps recommended at https://tesseract-ocr.github.io/tessdoc/tess4/TrainingTesseract-4.00.html#creating-training-data
  to install fonts and software packages.
* Clone the 4.1.1 release of Tesseract using the command:
  `git clone -b 4.1.1 --depth 1 https://github.com/tesseract-ocr/tesseract.git`
  The point of adding the -b flag and specifying the branch is to avoid getting the broken, most updated version of the 
  repository. 4.1.1 is the most recent stable release as of 7/9/2021.
* Find out which fonts are supported on your system. Record the supported fonts in a text file.
    * To do this, run the following command, as recommended at https://tesseract-ocr.github.io/tessdoc/Fonts.html:
      `text2image --find_fonts \
      --fonts_dir /usr/share/fonts \
      --text ./langdata/eng/eng.training_text \
      --min_coverage .9  \
      --outputbase ./langdata/eng/eng \
      |& grep raw \
      | sed -e 's/ :.*/@ \\/g' \
      | sed -e "s/^/  '/" \
      | sed -e "s/@/'/g" >./langdata/eng/fontslist.txt`
    * The purpose of doing this is to aid in the generation of images from fonts using tesstrain.sh. If you do not
      specify the supported fonts on your system, then tesstrain.sh will fail and tell you that it expected to find a
      training file associated with a font, but didn't.
    * You will want to use a Python script to reformat them so that they can be copied and pasted into the terminal.
      Split at newlines, at each newline take everything after the colon, and then join using spaces with double quotes as
      delimiters. This should be straightforward.
* Add a "best" directory in tesseract/tessdata, and use the command:
    `wget https://github.com/tesseract-ocr/tessdata_best/raw/master/ces.traineddata`
  To download the Czech traineddata file.
  The directory is called "best" because it uses floats instead of integers, making it slower and higher-quality than the
  integer-based model. (I use hand-wavy terms here because I do not fully understand.) Apparently only floats will permit
  fine-tuning.
* Follow the commands under "One-time Setup for Tesstutorial," replacing the directory ~/tesstutorial with the directory
  you prefer and replacing "eng" with "ces". Link: https://tesseract-ocr.github.io/tessdoc/tess4/TrainingTesseract-4.00.html#creating-training-data
  Do not re-clone the Github repo.
* In the langdata/ces directory, change the training text to be anything you like.
* `mv eng.user-patterns ces.user-patterns`
* `mv eng.user-words ces.user-words`
* Command used to create training data:
  `src/training/tesstrain.sh --fonts_dir /usr/share/fonts --lang eng --linedata_only   --noextract_font_properties --langdata_dir ../langdata   --tessdata_dir ./tessdata --output_dir ../train_plain_ces --fontlist "Abyssinica SIL" "Andale Mono" "AnjaliOldLipi" "Arial" "Arial Black" "Arial Bold" "Arial Bold Italic" "Arial Italic" "C059" "C059 Bold" "C059 Bold Italic" "C059 Italic" "Chilanka" "Comic Sans MS" "Comic Sans MS Bold" "Courier New" "Courier New Bold" "Courier New Bold Italic" "Courier New Italic" "D050000L" "DejaVu Math TeX Gyre" "DejaVu Sans" "DejaVu Sans Bold" "DejaVu Sans Bold Oblique" "DejaVu Sans Bold Oblique Semi-Condensed" "DejaVu Sans Bold Semi-Condensed" "DejaVu Sans Mono" "DejaVu Sans Mono Bold" "DejaVu Sans Mono Bold Oblique" "DejaVu Sans Mono Oblique" "DejaVu Sans Oblique" "DejaVu Sans Oblique Semi-Condensed" "DejaVu Sans Semi-Condensed" "DejaVu Sans Ultra-Light" "DejaVu Serif" "DejaVu Serif Bold" "DejaVu Serif Bold Italic" "DejaVu Serif Bold Italic Semi-Condensed" "DejaVu Serif Bold Semi-Condensed" "DejaVu Serif Italic" "DejaVu Serif Italic Semi-Condensed" "DejaVu Serif Semi-Condensed" "Dyuthi" "FreeMono" "FreeMono Bold" "FreeMono Bold Italic" "FreeMono Italic" "FreeSans" "FreeSans Italic" "FreeSans Semi-Bold" "FreeSans Semi-Bold Italic" "FreeSerif" "FreeSerif Bold" "FreeSerif Bold Italic" "FreeSerif Italic" "Garuda" "Garuda Bold" "Garuda Bold Oblique" "Garuda Oblique" "Gayathri" "Gayathri Bold" "Gayathri Thin" "Georgia" "Georgia Bold" "Georgia Bold Italic" "Georgia Italic" "Hack" "Hack Bold" "Hack Bold Italic" "Hack Italic" "Impact Condensed" "Karumbi" "Keraleeyam" "Khmer OS" "Khmer OS System" "Kinnari" "Kinnari Bold" "Kinnari Bold Italic" "Kinnari Bold Oblique" "Kinnari Italic" "Kinnari Oblique" "Laksaman" "Laksaman Bold" "Laksaman Bold Italic" "Laksaman Italic" "Liberation Mono" "Liberation Mono Bold" "Liberation Mono Bold Italic" "Liberation Mono Italic" "Liberation Sans" "Liberation Sans Bold" "Liberation Sans Bold Italic" "Liberation Sans Italic" "Liberation Sans Narrow Bold Condensed" "Liberation Sans Narrow Bold Italic Condensed" "Liberation Sans Narrow Condensed" "Liberation Sans Narrow Italic Condensed" "Liberation Serif" "Liberation Serif Bold" "Liberation Serif Bold Italic" "Liberation Serif Italic" "Loma" "Loma Bold" "Loma Bold Oblique" "Loma Oblique" "Manjari" "Manjari Bold" "Manjari Thin" "Meera" "Nakula" "Nimbus Mono PS" "Nimbus Mono PS Bold" "Nimbus Mono PS Bold Italic" "Nimbus Mono PS Italic" "Nimbus Roman" "Nimbus Roman, Bold" "Nimbus Roman, Bold Italic" "Nimbus Roman, Italic" "Nimbus Sans" "Nimbus Sans Bold" "Nimbus Sans Bold Italic" "Nimbus Sans Italic" "Nimbus Sans Narrow Bold Oblique Semi-Condensed" "Nimbus Sans Narrow Bold Semi-Condensed" "Nimbus Sans Narrow Oblique Semi-Condensed" "Nimbus Sans Narrow Semi-Condensed" "Norasi" "Norasi Bold" "Norasi Bold Italic" "Norasi Bold Oblique" "Norasi Italic" "Norasi Oblique" "Noto Mono" "Noto Sans CJK HK" "Noto Sans CJK HK Bold" "Noto Sans CJK JP" "Noto Sans CJK JP Bold" "Noto Sans CJK KR" "Noto Sans CJK KR Bold" "Noto Sans CJK SC" "Noto Sans CJK SC Bold" "Noto Sans CJK TC" "Noto Sans CJK TC Bold" "Noto Sans Mono CJK HK" "Noto Sans Mono CJK HK Bold" "Noto Sans Mono CJK JP" "Noto Sans Mono CJK JP Bold" "Noto Sans Mono CJK KR" "Noto Sans Mono CJK KR Bold" "Noto Sans Mono CJK SC" "Noto Sans Mono CJK SC Bold" "Noto Sans Mono CJK TC" "Noto Sans Mono CJK TC Bold" "Noto Serif CJK JP" "Noto Serif CJK JP Bold" "Noto Serif CJK KR" "Noto Serif CJK KR Bold" "Noto Serif CJK SC" "Noto Serif CJK SC Bold" "Noto Serif CJK TC" "Noto Serif CJK TC Bold" "P052" "P052 Bold" "P052 Bold Italic" "P052 Italic" "Padauk" "Padauk Bold" "Padauk Book" "Padauk Book, Bold" "Phetsarath OT" "Purisa" "Purisa Bold" "Purisa Bold Oblique" "Purisa Oblique" "Rachana" "Rasa" "Rasa Bold" "Rasa Light" "Rasa Medium" "Rasa Semi-Bold" "Sahadeva" "Sawasdee" "Sawasdee Bold" "Sawasdee Bold Oblique" "Sawasdee Oblique" "Standard Symbols PS" "Symbola Semi-Condensed" "Tibetan Machine Uni" "Times New Roman" "Times New Roman, Bold" "Times New Roman, Bold Italic" "Times New Roman, Italic" "Tlwg Mono" "Tlwg Mono Bold" "Tlwg Mono Bold Oblique" "Tlwg Mono Oblique" "Tlwg Typewriter" "Tlwg Typewriter Bold" "Tlwg Typewriter Bold Oblique" "Tlwg Typewriter Oblique" "Tlwg Typist" "Tlwg Typist Bold" "Tlwg Typist Bold Oblique" "Tlwg Typist Oblique" "Tlwg Typo" "Tlwg Typo Bold" "Tlwg Typo Bold Oblique" "Tlwg Typo Oblique" "Trebuchet MS" "Trebuchet MS Bold" "Trebuchet MS Bold Italic" "Trebuchet MS Italic" "URW Bookman Light" "URW Bookman Light Italic" "URW Bookman Semi-Bold" "URW Bookman Semi-Bold Italic" "URW Gothic" "URW Gothic Oblique" "URW Gothic Semi-Bold" "URW Gothic Semi-Bold Oblique" "Ubuntu" "Ubuntu Bold" "Ubuntu Bold Italic" "Ubuntu Condensed" "Ubuntu Italic" "Ubuntu Light" "Ubuntu Light Italic" "Ubuntu Medium" "Ubuntu Medium Italic" "Ubuntu Mono" "Ubuntu Mono Bold" "Ubuntu Mono Bold Italic" "Ubuntu Mono Italic" "Ubuntu weight=255" "Umpush" "Umpush Bold" "Umpush Bold Oblique" "Umpush Light" "Umpush Light Oblique" "Umpush Oblique" "Uroob" "Verdana" "Verdana Bold" "Verdana Bold Italic" "Verdana Italic" "Waree" "Waree Bold" "Waree Bold Oblique" "Waree Oblique" "Webdings" "Yrsa" "Yrsa Bold" "Yrsa Light" "Yrsa Medium" "Yrsa Semi-Bold" "Z003 Medium Italic" "padmaa-Bold.1.1 Bold" --lang ces`
* In the tesseract directory, run the following:
  * ./autogen.sh
  * ./configure
  * make
  * make training
  * sudo make training-install


  src/training/lstmtraining --model_output ../train_plain_ces/plain_ces \
  --net_spec '[1,36,0,1 Ct3,3,16 Mp3,3 Lfys48 Lfx96 Lrx96 Lfx256 O1c111]' \
  --traineddata ~/Documents/awca/awca-ocr/tesstrain/train_plain_ces/ces/ces.traineddata \
  --old_traineddata tessdata/best/ces.traineddata \
  --train_listfile ~/Documents/awca/awca-ocr/tesstrain/train_plain_ces/ces.training_files.txt \
  --max_iterations 3600

* Run the following (changing directory names as appropriate) to get the LSTM file for training:
  training/combine_tessdata \
  -e ~/Documents/awca/awca-ocr/tesstrain/tesseract/tessdata/best/ces.traineddata \
  ~/Documents/awca/awca-ocr/tesstrain/train_plain_ces_from_full/ces.lstm

* The following will create the training files that we actually want. Note that it is necessary to first replace the contents of langdata/ces/ces.training_text with the desired training text.
  src/training/tesstrain.sh \
  --fonts_dir /usr/share/fonts \
  --lang ces \
  --linedata_only \
  --noextract_font_properties \
  --langdata_dir ../langdata \
  --tessdata_dir ./tessdata \
  --output_dir ../translit \
  --fontlist "Abyssinica SIL" "Andale Mono" "AnjaliOldLipi" "Arial" "Arial Black" "Arial Bold" "Arial Bold Italic" "Arial Italic" "C059" "C059 Bold" "C059 Bold Italic" "C059 Italic" "Chilanka" "Comic Sans MS" "Comic Sans MS Bold" "Courier New" "Courier New Bold" "Courier New Bold Italic" "Courier New Italic" "D050000L" "DejaVu Math TeX Gyre" "DejaVu Sans" "DejaVu Sans Bold" "DejaVu Sans Bold Oblique" "DejaVu Sans Bold Oblique Semi-Condensed" "DejaVu Sans Bold Semi-Condensed" "DejaVu Sans Mono" "DejaVu Sans Mono Bold" "DejaVu Sans Mono Bold Oblique" "DejaVu Sans Mono Oblique" "DejaVu Sans Oblique" "DejaVu Sans Oblique Semi-Condensed" "DejaVu Sans Semi-Condensed" "DejaVu Sans Ultra-Light" "DejaVu Serif" "DejaVu Serif Bold" "DejaVu Serif Bold Italic" "DejaVu Serif Bold Italic Semi-Condensed" "DejaVu Serif Bold Semi-Condensed" "DejaVu Serif Italic" "DejaVu Serif Italic Semi-Condensed" "DejaVu Serif Semi-Condensed" "Dyuthi" "FreeMono" "FreeMono Bold" "FreeMono Bold Italic" "FreeMono Italic" "FreeSans" "FreeSans Italic" "FreeSans Semi-Bold" "FreeSans Semi-Bold Italic" "FreeSerif" "FreeSerif Bold" "FreeSerif Bold Italic" "FreeSerif Italic" "Garuda" "Garuda Bold" "Garuda Bold Oblique" "Garuda Oblique" "Gayathri" "Gayathri Bold" "Gayathri Thin" "Georgia" "Georgia Bold" "Georgia Bold Italic" "Georgia Italic" "Hack" "Hack Bold" "Hack Bold Italic" "Hack Italic" "Impact Condensed" "Karumbi" "Keraleeyam" "Khmer OS" "Khmer OS System" "Kinnari" "Kinnari Bold" "Kinnari Bold Italic" "Kinnari Bold Oblique" "Kinnari Italic" "Kinnari Oblique" "Laksaman" "Laksaman Bold" "Laksaman Bold Italic" "Laksaman Italic" "Liberation Mono" "Liberation Mono Bold" "Liberation Mono Bold Italic" "Liberation Mono Italic" "Liberation Sans" "Liberation Sans Bold" "Liberation Sans Bold Italic" "Liberation Sans Italic" "Liberation Sans Narrow Bold Condensed" "Liberation Sans Narrow Bold Italic Condensed" "Liberation Sans Narrow Condensed" "Liberation Sans Narrow Italic Condensed" "Liberation Serif" "Liberation Serif Bold" "Liberation Serif Bold Italic" "Liberation Serif Italic" "Loma" "Loma Bold" "Loma Bold Oblique" "Loma Oblique" "Manjari" "Manjari Bold" "Manjari Thin" "Meera" "Nakula" "Nimbus Mono PS" "Nimbus Mono PS Bold" "Nimbus Mono PS Bold Italic" "Nimbus Mono PS Italic" "Nimbus Roman" "Nimbus Roman, Bold" "Nimbus Roman, Bold Italic" "Nimbus Roman, Italic" "Nimbus Sans" "Nimbus Sans Bold" "Nimbus Sans Bold Italic" "Nimbus Sans Italic" "Nimbus Sans Narrow Bold Oblique Semi-Condensed" "Nimbus Sans Narrow Bold Semi-Condensed" "Nimbus Sans Narrow Oblique Semi-Condensed" "Nimbus Sans Narrow Semi-Condensed" "Norasi" "Norasi Bold" "Norasi Bold Italic" "Norasi Bold Oblique" "Norasi Italic" "Norasi Oblique" "Noto Mono" "Noto Sans CJK HK" "Noto Sans CJK HK Bold" "Noto Sans CJK JP" "Noto Sans CJK JP Bold" "Noto Sans CJK KR" "Noto Sans CJK KR Bold" "Noto Sans CJK SC" "Noto Sans CJK SC Bold" "Noto Sans CJK TC" "Noto Sans CJK TC Bold" "Noto Sans Mono CJK HK" "Noto Sans Mono CJK HK Bold" "Noto Sans Mono CJK JP" "Noto Sans Mono CJK JP Bold" "Noto Sans Mono CJK KR" "Noto Sans Mono CJK KR Bold" "Noto Sans Mono CJK SC" "Noto Sans Mono CJK SC Bold" "Noto Sans Mono CJK TC" "Noto Sans Mono CJK TC Bold" "Noto Serif CJK JP" "Noto Serif CJK JP Bold" "Noto Serif CJK KR" "Noto Serif CJK KR Bold" "Noto Serif CJK SC" "Noto Serif CJK SC Bold" "Noto Serif CJK TC" "Noto Serif CJK TC Bold" "P052" "P052 Bold" "P052 Bold Italic" "P052 Italic" "Padauk" "Padauk Bold" "Padauk Book" "Padauk Book, Bold" "Phetsarath OT" "Purisa" "Purisa Bold" "Purisa Bold Oblique" "Purisa Oblique" "Rachana" "Rasa" "Rasa Bold" "Rasa Light" "Rasa Medium" "Rasa Semi-Bold" "Sahadeva" "Sawasdee" "Sawasdee Bold" "Sawasdee Bold Oblique" "Sawasdee Oblique" "Standard Symbols PS" "Symbola Semi-Condensed" "Tibetan Machine Uni" "Times New Roman" "Times New Roman, Bold" "Times New Roman, Bold Italic" "Times New Roman, Italic" "Tlwg Mono" "Tlwg Mono Bold" "Tlwg Mono Bold Oblique" "Tlwg Mono Oblique" "Tlwg Typewriter" "Tlwg Typewriter Bold" "Tlwg Typewriter Bold Oblique" "Tlwg Typewriter Oblique" "Tlwg Typist" "Tlwg Typist Bold" "Tlwg Typist Bold Oblique" "Tlwg Typist Oblique" "Tlwg Typo" "Tlwg Typo Bold" "Tlwg Typo Bold Oblique" "Tlwg Typo Oblique" "Trebuchet MS" "Trebuchet MS Bold" "Trebuchet MS Bold Italic" "Trebuchet MS Italic" "URW Bookman Light" "URW Bookman Light Italic" "URW Bookman Semi-Bold" "URW Bookman Semi-Bold Italic" "URW Gothic" "URW Gothic Oblique" "URW Gothic Semi-Bold" "URW Gothic Semi-Bold Oblique" "Ubuntu" "Ubuntu Bold" "Ubuntu Bold Italic" "Ubuntu Condensed" "Ubuntu Italic" "Ubuntu Light" "Ubuntu Light Italic" "Ubuntu Medium" "Ubuntu Medium Italic" "Ubuntu Mono" "Ubuntu Mono Bold" "Ubuntu Mono Bold Italic" "Ubuntu Mono Italic" "Ubuntu weight=255" "Umpush" "Umpush Bold" "Umpush Bold Oblique" "Umpush Light" "Umpush Light Oblique" "Umpush Oblique" "Uroob" "Verdana" "Verdana Bold" "Verdana Bold Italic" "Verdana Italic" "Waree" "Waree Bold" "Waree Bold Oblique" "Waree Oblique" "Webdings" "Yrsa" "Yrsa Bold" "Yrsa Light" "Yrsa Medium" "Yrsa Semi-Bold" "Z003 Medium Italic" "padmaa-Bold.1.1 Bold"
* Warning: The training files take days to create, even with several threads working on all available cores, and they
  consume an enormous amount of space. For reference, when I used the ur-iii text, which is on the order of 14 MB
  (something like that), it was going to use roughly 45 GB of disk space in the /tmp/ directory, and so I had to kill the
  process. Going forward I will break the text into chunks (eighths).

The following will create the eval files that we want. Before running it, you will have to swap out the sample text file at
tesstrain/langdata/ces/ces.training_text.
IMPORTANT: I used tesstrain/langdata/ces/ur-iii/training_text_7.txt as the eval file. Of course, training_text_7.txt can still be
used for training, but if it is, then of course our evaluation data will become meaningless due to overfitting.
src/training/tesstrain.sh \
  --fonts_dir /usr/share/fonts \
  --lang ces \
  --linedata_only \
  --noextract_font_properties \
  --langdata_dir ../langdata \
  --tessdata_dir ./tessdata \
  --output_dir ../translit_eval \
  --fontlist "Arial" "Arial Bold" "Arial Italic" \
    "Courier New" "Courier New Italic" \
    "Georgia" "Georgia Bold" "Georgia Italic" \
    "Times New Roman" "Times New Roman, Bold" "Times New Roman, Italic" \
    "Verdana" "Verdana Italic"

* The following command will create the model that we actually want:
  src/training/lstmtraining \
  --model_output ../train_translit/checkpoints/translit \
  --continue_from ~/Documents/awca/awca-ocr/tesstrain/train_translit/checkpoints/ces.lstm \
  --traineddata ~/Documents/awca/awca-ocr/tesstrain/train_translit/ces/ces.traineddata \
  --old_traineddata tessdata/best/ces.traineddata \
  --train_listfile ~/Documents/awca/awca-ocr/tesstrain/train_translit/ces.training_files.txt \
  --max_iterations 3600

  sudo training/lstmtraining --stop_training \
  --continue_from ~/Documents/awca/awca-ocr/tesstrain/train_plain_ces/plain_ces_checkpoint \
  --traineddata ~/Documents/awca/awca-ocr/tesstrain/train_plain_ces/ces/ces.traineddata \
  --model_output /usr/local/share/tessdata/ces2.traineddata
