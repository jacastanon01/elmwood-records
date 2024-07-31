from genericpath import isfile
from math import inf
import cv2
import pytesseract
import os

from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError
from process_image import get_form_page

CURRENT_DIR = os.getcwd()


def read_coordinates(
    file_path: str,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        # print(lines)
        coords = []
        for line in lines:
            line = line.strip()
            # print(line.split(" "))
            x1, y1, x2, y2 = line.split(" ")
            coords.append(((float(x1), float(y1)), (float(x2), float(y2))))
        return [(coords[0], coords[1]), (coords[2], coords[3])]
    except FileNotFoundError:
        print("File not found!")
        exit()


def crop_segment(image, start, end):
    x1, y1 = start
    x2, y2 = end
    print(start, end, sep="\n")
    return image[y1:y2, x1:x2]


def preprocess_segment(segment, i: int):
    gray = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f"segment-{i}.jpg", segment)
    return thresh


def process_image(pdf_path: str):
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

    image = get_form_page(pages)
    return image


def extract_text(segment):
    my_config = r"--oem 3 --psm 6"  # 3 4 6 11
    text = pytesseract.image_to_string(segment, lang="eng", config=my_config)
    return text


def select_files(directory: str, max_files=10) -> list[str]:
    dir = f"{CURRENT_DIR}/{directory}"
    # if not os.path.isfile(dir):
    #     select_files(dir)
    all_files = os.listdir(dir)
    files = [os.path.join(dir, f) for f in all_files[:max_files]]
    return files


def main():
    unprocessed = (
        "/Users/jacobcastanon/workspace/projects/elmwood-records/images/name.jpg"
    )
    processed = (
        "/Users/jacobcastanon/workspace/projects/elmwood-records/images/threshold.jpg"
    )

    files = select_files("Cards/CO-DAR")
    # images = [process_image(file) for file in files]
    coords = read_coordinates("ref_points.txt")

    segments = [
        crop_segment("images/im_bw.jpg", start, end)
        for i, (start, end) in enumerate(coords)
    ]
    process = [preprocess_segment(segment, i) for i, segment in enumerate(segments)]
    text = [extract_text(segment) for segment in process]
    print(text[0])


if __name__ == "__main__":
    main()
    # c = read_coordinates("ref_points.txt")
    # print(c[0])
    # files = select_files("Cards/CO-DAR")
    # images = [process_image(file) for file in files]

    # for start, end in co:
    #     print(start, end, sep=", ")
    # print("------")
