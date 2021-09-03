import os
import fitz
import time


def pdf_from_images(image_dir, output_file='./images.pdf'):
    """Creates a PDF containing all of the images in `image_dir`, in
    alphabetical order by file name.
    Adapted from https://pymupdf.readthedocs.io/en/latest/faq.html
    :param image_dir: The directory containing all images to be
        included in the PDF
    :param output_file: The path to the file to be created
    """
    out = fitz.open()
    image_files = os.listdir(image_dir)
    image_files.sort()
    t0 = time.time()
    for i, image_file in enumerate(image_files):
        image = fitz.open(os.path.join(image_dir, image_file))
        image_pdf = fitz.open('pdf', image.convert_to_pdf())
        rect = image[0].rect
        page = out.new_page(width=rect.width, height=rect.height)
        page.show_pdf_page(rect, image_pdf, 0)
        print(
            '{} images copied in {:.0f} seconds. {:.1f}% complete.'.format(
                i + 1, time.time() - t0, (i + 1) / len(image_files)
            ),
            end='\r'
        )
    print('Saving PDF as {}...'.format(output_file))
    out.save(output_file)
