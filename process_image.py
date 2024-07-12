import numpy as np
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

    return cv2.countNonZero(image) / (height * width)


def process_image(pdf_path: str):
    # -c tessedit_char_whitelist=\'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-/., "

    my_config = r"--oem 3 --psm 3"  # 3 4 6 11

    pages = convert_from_path(
        pdf_path,
        300,
        grayscale=True,
    )

    first_page_pixels, second_page_pixels = [np.array(p, np.uint8) for p in pages]

    first_white = calulcate_white_px(first_page_pixels)
    second_white = calulcate_white_px(second_page_pixels)

    page = pages[0] if first_white < second_white else pages[1]

    im_path = generate_jpg_file("name", page)
    image = cv2.imread(im_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    assert image is not None

    im_bw = adapt_thresh(image)
    # threshold, im_bw = cv2.threshold(
    #     image, 210, 245, cv2.THRESH_BINARY
    # )  # cv2.THRESH_OTSU

    cv2.imwrite("images/im_bw.jpg", im_bw)
    ntext = pytesseract.image_to_string(im_bw, lang="eng", config=my_config)
    print(ntext)
    cv2.destroyAllWindows()


def adapt_thresh(image):
    "Manipulates image to be more readable to OCR"
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    image = image.copy()

    image = cv2.bitwise_not(image)
    image = cv2.fastNlMeansDenoising(image, h=40)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    cv2.imwrite("images/blur.jpg", image)
    image = cv2.dilate(image, kernel_dilate, iterations=1)
    image = cv2.erode(image, kernel_erode, iterations=1)
    image = cv2.bitwise_not(image)
    image = cv2.adaptiveThreshold(
        image, 240, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5
    )

    cv2.imwrite("images/threshold.jpg", image)
    return image


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


# def draw_boundary_boxes(image):
#     base_image = image.copy()

#     contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = sorted(
#         (contours[0] if len(contours) == 2 else contours[1]),
#         key=lambda x: cv2.boundingRect(x)[0],
#     )

#     for idx, c in enumerate(cnts):
#         print(c)
#         if len(c) < 2:
#             continue
#         x, y = cv2.boundingRect(c)


#         print(f"Bounding box {idx}: x={x}, y={y}")

#         roi = image[y, x]
#         draw_points = (x, y)  # tells where on the image to draw the boxes
#         cv2.rectangle(base_image, (x, y), draw_points, (0, 255, 0), 2)
#         cv2.imwrite(f"images/roi{idx}.jpg", roi)

#     cv2.imwrite("images/bbox.jpg", base_image)
#     return base_image


# def remove_noise(image):
#     image = cv2.bitwise_not(image)
#     kernel = np.ones([2, 2], np.uint8)
#     image = cv2.dilate(image, kernel, iterations=1)
#     kernel = np.ones([1, 1], np.uint8)
#     image = cv2.erode(image, kernel, iterations=1)
#     image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
#     image = cv2.medianBlur(image, 3)
#     image = cv2.bitwise_not(image)
#     return image


if __name__ == "__main__":
    process_image("test_remarks.pdf")
