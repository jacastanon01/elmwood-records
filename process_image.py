import numpy as np
import cv2
import os
from pdf2image import convert_from_path
import pytesseract


def calulcate_white_px(image) -> int:
    """Calculates number of white pixels in image

    Args:
        image (Numpy Array):

    Returns:
        int: percentage of white px/total px
    """
    width, height = image.shape

    return cv2.countNonZero(image) / (height * width)


def process_image(pdf_path: str) -> str:
    """Takes pdf filepath and converts to image to perform processing

    Args:
        pdf_path (str): path... to the pdf file

    Returns:
        text (str): result of tesserect OCR conversion to string
    """
    my_config = r"--oem 3 --psm 6"  # 3 4 6 11

    pages = convert_from_path(
        pdf_path,
        300,
        grayscale=True,
    )

    first_page_pixels, second_page_pixels = [np.array(p, np.uint8) for p in pages]

    first_white = calulcate_white_px(first_page_pixels)
    second_white = calulcate_white_px(second_page_pixels)

    page = pages[0] if first_white < second_white else pages[1]

    save_page_path = "images/name.jpg"

    page.save(save_page_path, "JPEG")

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
    return text


def adapt_thresh(image):
    "Manipulates image to be more readable to OCR"
    # kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    # kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    image = image.copy()

    image = cv2.bitwise_not(image)
    image = cv2.fastNlMeansDenoising(image, h=60)
    image = cv2.GaussianBlur(image, (7, 7), 0)
    cv2.imwrite("images/blur.jpg", image)
    # image = cv2.dilate(image, kernel_dilate, iterations=1)
    # image = cv2.erode(image, kernel_erode, iterations=1)
    image = cv2.bitwise_not(image)
    image = cv2.adaptiveThreshold(
        image, 240, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3
    )

    cv2.imwrite("images/threshold.jpg", image)
    return image


if __name__ == "__main__":
    text = process_image("test_lines.pdf")
    print(text)
