import numpy as np
import cv2
import os
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import PDFPageCountError
from PIL import Image as PILImage


def get_form_page(pages):
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


def adapt_thresh(image):
    "Manipulates image to be more readable to OCR"
    # kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

    image = image.copy()

    image = cv2.bitwise_not(image)
    image = cv2.fastNlMeansDenoising(image, h=40)
    # image = cv2.GaussianBlur(image, (3,3), 3)
    cv2.imwrite("images/blur.jpg", image)
    # image = cv2.dilate(image, kernel_dilate, iterations=1)
    image = cv2.bitwise_not(image)
    image = cv2.adaptiveThreshold(
        image, 240, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3
    )
    image = cv2.erode(image, kernel_erode, iterations=1)

    cv2.imwrite("images/threshold.jpg", image)
    return image


def process_image(pdf_path: str) -> str:
    """Takes pdf filepath and converts to image to perform processing

    Args:
        pdf_path (str): path... to the pdf file

    Returns:
        text (str): result of tesserect OCR conversion to string
    """

    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        return ""

    my_config = r"--oem 3 --psm 6"  # 3 4 6 11

    try:
        pages = convert_from_path(pdf_path, 300)
    except PDFPageCountError:
        print(f"Unable to get page count for file: {pdf_path}")
        return ""
    except Exception as e:
        print(f"An error occurred while converting the PDF: {e}")
        return ""

    page = get_form_page(pages)

    save_page_path = "images/name.jpg"

    # page.save(save_page_path, "JPEG")
    cv2.imwrite(save_page_path, page)

    try:
        image = cv2.imread(save_page_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        im_bw = adapt_thresh(image)
        # threshold, im_bw = cv2.threshold(
        #     image, 210, 245, cv2.THRESH_BINARY
        # )  # cv2.THRESH_OTSU

        cv2.imwrite("images/im_bw.jpg", im_bw)
        text = pytesseract.image_to_string(im_bw, lang="eng", config=my_config)
        # print(text)
        cv2.destroyAllWindows()
        print("processed...")
        return text
    except Exception as e:
        print(f"An error occurred during image processing: {e}")
        return ""


def other(file: str):
    with open(file, "rb") as f:
        images = convert_from_bytes(f.read(), 500)
        page = get_form_page(images)
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        filename = "{}.jpg".format(os.getpid())
        edges = cv2.Canny(gray, 0, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
        )

        # Draw lines on the image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(page, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite("lines_detected.jpg", page)
