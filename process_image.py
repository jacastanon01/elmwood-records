import enum
import itertools
from typing import Tuple, final
import numpy as np
from PIL import Image, ImageChops
import cv2
import os
from pdf2image import convert_from_path
import pytesseract


def calulcate_white_px(image) -> int:
    """Calculates numnber of white pixels in image

    Args:
        image (Numpy Array):

    Returns:
        int: percentage of white px/total px
    """
    width, height = image.shape

    _, im_bw = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(im_bw) / (height * width)


def process_image(pdf_path: str):
    bw_path = "images/im_bw.jpg"
    # -c tessedit_char_whitelist=\'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-/., "

    my_config = r"--oem 3 --psm 3"  # 3 4 6 11

    pages = convert_from_path(
        pdf_path,
        300,
        # single_file=True,
        grayscale=True,
    )

    first_page_pixels, second_page_pixels = [np.array(p, np.uint8) for p in pages]

    first_white = calulcate_white_px(first_page_pixels)
    second_white = calulcate_white_px(second_page_pixels)

    page = pages[0] if first_white < second_white else pages[1]
    # page.save("images/page.jpg", "JPEG")

    im_path = generate_jpg_file("name", page)
    image = cv2.imread(im_path)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    assert image is not None
    # cv2.imwrite("images/page.jpg", image)

    # box_img = pytesseract.image_to_string(image_boxes, lang="eng", config=my_config)
    threshold, im_bw = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # noise = remove_noise(image)
    # thresh = adapt_thresh(image)
    image_boxes = draw_boundary_boxes(im_bw)

    # im_bw = cv2.dilate(im_bw, np.ones((2, 2), np.uint8), iterations=1)
    # im_bw = cv2.erode(im_bw, np.ones((1, 1), np.uint8), iterations=1)

    # print(box_img)

    # cv2.imwrite(bw_path, )

    for i in (image_boxes, im_bw):
        ntext = pytesseract.image_to_string(i, lang="eng", config=my_config)
        print(f"Final Text {ntext}", end="\n\n\n")

    # contours, _ = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.destroyAllWindows()


def is_image_blank2(image, threshold=5):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = remove_noise(image)

    _, binary = cv2.threshold(
        image, 240, 255, cv2.THRESH_BINARY
    )  # Apply a binary threshold
    binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    non_zero_pixels = cv2.countNonZero(binary)
    height, width = binary.shape
    total_pixels = height * width
    # Check if the number of non-zero pixels is below the threshold
    return non_zero_pixels < threshold


def generate_jpg_file(filename: str, image) -> str:
    """creates images folder if one doesn't exist and generates jpg file

    Args:
        filename (str): Name of jpg file
        image (Image): PIL Image Object to save to file

    Returns:
        str: absolute path to generated jpg
    """
    dirpath = f"{os.getcwd()}/images"

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    abs_image_path = os.path.join(f"{dirpath}/{filename}.jpg")
    if os.path.exists(abs_image_path):
        os.remove(abs_image_path)
        print(f"{filename.split("/")[-1]} deleted")
    image.save(abs_image_path, "JPEG")
    return abs_image_path


def draw_boundary_boxes(image):
    # ksize = (4, 2)  # (8,3) (11,5) (13,5)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    # kernel = np.ones((1, 1), np.uint8)

    base_image = image.copy()
    # im_h, im_w = image.shape
    # img = cv2.GaussianBlur(image, (7, 7), 0)
    # cv2.imwrite("images/blur.jpg", img)

    # img = adapt_thresh(base_image)

    # img = cv2.dilate(image, kernel, iterations=1)
    # kernel = np.ones((2, 2), np.uint8)
    # img = cv2.erode(img, kernel, iterations=1)
    # cv2.imwrite("images/dilate.jpg", img)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(
        (contours[0] if len(contours) == 2 else contours[1]),
        key=lambda x: cv2.boundingRect(x)[0],
    )

    for idx, c in enumerate(cnts):
        print(c)
        if len(c) < 2:
            continue
        x, y = cv2.boundingRect(c)

        # print(f"h: {h}\nw: {w}")
        print(f"Bounding box {idx}: x={x}, y={y}")
        # if h > 30 and w > 70:
        roi = image[y, x]
        draw_points = (x, y)  # tells where on the image to draw the boxes
        cv2.rectangle(base_image, (x, y), draw_points, (0, 255, 0), 2)
        # if w > 50 and h > 20:
        cv2.imwrite(f"images/roi{idx}.jpg", roi)

    cv2.imwrite("images/bbox.jpg", base_image)
    return base_image


def remove_noise(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones([2, 2], np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones([1, 1], np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    image = cv2.bitwise_not(image)
    return image


def adapt_thresh(image):
    # print("Hello from threshold")
    # if len(image.shape) == 3:
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.bitwise_not(image)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    image = image.copy()

    image = cv2.bitwise_not(image)
    image = cv2.fastNlMeansDenoising(image, h=50)
    image = cv2.GaussianBlur(image, (7, 7), 0)
    cv2.imwrite("images/blur.jpg", image)
    image = cv2.dilate(image, kernel_dilate, iterations=2)
    image = cv2.erode(image, kernel_erode, iterations=1)
    image = cv2.bitwise_not(image)
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 3
    )

    cv2.imwrite("images/threshold.jpg", image)
    return image


if __name__ == "__main__":
    process_image("test_lines.pdf")
