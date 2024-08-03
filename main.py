from itertools import batched
import cv2
import pytesseract
import os

from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError
from process_image import get_form_page

CURRENT_DIR = os.getcwd()


def read_coordinates(
    file_path: str,
) -> list[tuple[float, ...]]:
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        # print(lines)
        coords = []
        group_tuples = []
        # print(lines)
        for line in lines:
            line = line.strip()
            # print(line.split(" "))
            # print(line)
            group_tuples = list(batched((float(i) for i in line.split(" ")), 2))
            # print(group_tuples)
            x1, y1, x2, y2 = line.split(" ")
            coords.append(group_tuples)
        # print(coords)
        # return tuple((coords[0], coords[1])), tuple((coords[2], coords[3]))
        return coords
    except FileNotFoundError:
        print("File not found!")
        exit()


def crop_segment(image, start, end):
    x1, y1 = start
    x2, y2 = end
    print(start, end, x1, y1, sep="\n")
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
    images = [process_image(file) for file in files]
    cv2.imwrite("image.jpg", images[0])
    # ? convert np array into image to pass to functions
    # points = read_coordinates("ref_points.txt")
    # # print(points)

    # # for p in points:
    # #     print(p, end="\n------\n")
    # segments = [crop_segment(images[0], start, end) for (start, end) in points]
    # process = [preprocess_segment(segment, i) for i, segment in enumerate(segments)]
    # text = [extract_text(segment) for segment in process]
    # print(text)


if __name__ == "__main__":
    main()
    # c = read_coordinates("ref_points.txt")
    # print(c[0])
    # files = select_files("Cards/CO-DAR")
    # images = [process_image(file) for file in files]

    # for start, end in co:
    #     print(start, end, sep=", ")
    # print("------")
