import argparse
import os

from tesseract_manager import Text

parser = argparse.ArgumentParser(description="Run ocr on every 10th document, starting with document i.")
parser.add_argument("i", help="The zero-based index of the first document to be OCR'd.")

args = parser.parse_args()

print(args.i)

with open("./index.txt", "r") as f:
    pdfs = f.read().split("\n")

processed_pdfs = set()

for processed_pdfs_list in os.listdir():
    if "processed_pdfs" in processed_pdfs_list:
        with open(f"./{processed_pdfs_list}", "r") as f:
            for pdf in f.read().split("\n"):
                processed_pdfs.add(pdf)

for idx, pdf in enumerate(pdfs):
    if idx % 10 == int(args.i) and pdf not in processed_pdfs:
        print(idx, pdf)
        if not os.path.isfile(pdf):
            print(f"WARNING: \"{pdf}\" does not exist.")
            continue
        Text(pdf, f"../ocr-output/{idx}_{os.path.basename(pdf).split('.')[0]}").save_ocr()
        with open(f"processed_pdfs_list_{args.i}.txt", "a") as f:
            f.write(pdf + "\n")
    
