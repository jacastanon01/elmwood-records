from itertools import batched
import cv2
import pytesseract
import os
import numpy as np

from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError
from PIL import Image as PILImage

CURRENT_DIR = os.getcwd()


def remove_blank_page(pages):
    """Takes in a list of Images and determines which image contains the record with the burial data

    Args:
        pages (list[Image]): array of images returned from pdf2image convert function

    Returns:
        Image: Returns page determined to not be blank
    """

    if len(pages) < 2:
        page = pages[0]
    else:
        first_page_pixels, second_page_pixels = [np.array(p, np.uint8) for p in pages]
        first_white = calulcate_white_px(first_page_pixels)
        second_white = calulcate_white_px(second_page_pixels)
        page = pages[0] if first_white < second_white else pages[1]

    if isinstance(page, PILImage.Image):
        page = np.array(page, np.uint8)

    return page


def calulcate_white_px(image) -> int:
    """Calculates number of white pixels in image

    Args:
        image (Numpy Array):

    Returns:
        int: percentage of white px/total px
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape

    return cv2.countNonZero(image) / (height * width)


def read_coordinates(
    file_path: str,
) -> list[tuple[float, ...]]:
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        coords = []
        group_tuples = []
        for line in lines:
            line = line.strip()
            group_tuples = list(batched((float(i) for i in line.split(" ")), 2))
            coords.append(group_tuples)
        return coords
    except FileNotFoundError:
        print("File not found!")
        exit()


def crop_segment(image, start, end):
    # x1, y1 = ((int(x), int(y)) for x, y in start)
    # x2, y2 = ((int(x), int(y)) for x, y in end)
    x1, y1 = map(int, start)
    x2, y2 = map(int, end)
    return image[y1:y2, x1:x2]


def preprocess_segment(segment, i: int):
    gray = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f"segment-{i}.jpg", segment)
    return thresh


def convert_pdf_to_image(pdf_path: str):
    """Takes pdf filepath and converts to image to perform processing

    Args:
        pdf_path (str): path... to the pdf file

    Returns:
        text (str): result of tesserect OCR conversion to string
    """
    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")

    try:
        pages = convert_from_path(pdf_path, 300)
    except PDFPageCountError:
        print(f"Unable to get page count for file: {pdf_path}")
    except Exception as e:
        print(f"An error occurred while converting the PDF: {e}")

    image = remove_blank_page(pages)
    return image, pdf_path[-9:-5]


def extract_text(segment):
    my_config = r"--oem 3 --psm 11"  # 3 4 6 11
    text = pytesseract.image_to_string(segment, lang="eng", config=my_config)
    text = text.replace("\n", "")
    return text


def select_files(directory: str, max_files=10) -> list[str]:
    dir = f"{CURRENT_DIR}/{directory}"
    all_files = os.listdir(dir)
    files = [os.path.join(dir, f) for f in all_files[:max_files]]
    return files
