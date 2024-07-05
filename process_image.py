from typing import final
import numpy as np
import cv2
import os
from pdf2image import convert_from_path
import pytesseract


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


def align_text(image):
    max_features = 500
    template = cv2.imread("grey.jpg")

    temp_grey = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    img_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB.create()

    # keypoint1, descriptor1 = orb.compute(temp_grey)


def draw_boundary_boxes(image):
    ksize = (12, 5)  # (8,3) (11,5) (13,5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)

    base_image = image.copy()
    # im_h, im_w = image.shape
    img = cv2.GaussianBlur(image, (7, 7), 0)
    cv2.imwrite("images/blur.jpg", img)

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite("images/kernel.jpg", kernel)

    img = cv2.dilate(img, kernel, iterations=1)
    cv2.imwrite("images/dilate.jpg", img)

    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(
        (contours[0] if len(contours) == 2 else contours[1]),
        key=lambda x: cv2.boundingRect(x)[0],
    )

    for c in cnts:
        if len(c) < 5:
            continue
        x, y, w, h = cv2.boundingRect(c)

        if h < 10 and w > 40:
            roi = base_image[y : y + h, x : x + w]
            draw_points = (x + w, y + h)  # tells where on the image to draw the boxes
            cv2.rectangle(image, (x, y), draw_points, (0, 255, 0), 2)
            # if w > 50 and h > 20:
            cv2.imwrite(f"images/roi.jpg", roi)

    cv2.imwrite("images/bbox.jpg", image)
    return image


def process_image():
    bw_path = "images/im_bw.jpg"
    no_noise_path = "images/no_noise.jpg"
    # -c tessedit_char_whitelist=\'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-/., "

    my_config = r"--oem 3 --psm 11"  # 3 4 6 11
    pdf_path = "test_lines.pdf"

    page = convert_from_path(pdf_path, 300, single_file=True, grayscale=True)[0]
    im_path = generate_jpg_file("page", page)

    image = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(im_path, image)

    image_boxes = draw_boundary_boxes(image)

    text = pytesseract.image_to_string(image_boxes, lang="eng", config=my_config)

    print(text)
    return
    inverse = cv2.bitwise_not(image)
    cv2.imwrite("inverse.jpg", inverse)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("grey.jpg", image)

    # kernel = np.ones((5, 5), np.uint8)
    # erosion = cv2.erode(grey, kernel, iterations=1)
    # cv2.imwrite("erosion.jpg", erosion)

    threshold, im_bw = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    cv2.imwrite(bw_path, im_bw)

    # image = cv2.GaussianBlur(im_bw, (7, 7), 0)
    # cv2.imwrite("images/blur.jpg", image)

    # cv2.imwrite(bw_path, im_bw)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
    # cv2.imwrite("images/kernel.jpg", kernel)

    # dilate = cv2.dilate(image, kernel, iterations=1)
    # cv2.imwrite("images/ddddilate.jpg", dilate)

    # contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = sorted(
    #     (contours[0] if len(contours) == 2 else contours[1]),
    #     key=lambda x: cv2.boundingRect(x)[0],
    # )

    # for c in cnts:
    #     x, y, w, h = cv2.boundingRect(c)
    #     draw_points = (x + w, y + h)  # tells where on th eimage to draw the box
    #     cv2.rectangle(im_bw, (x, y), draw_points, (0, 255, 0), 2)

    # cv2.imwrite("images/bbox.jpg", im_bw)
    # return
    # im_bw = adapt_thresh(image)

    # im_bw = adapt_thresh(image)
    # text = pytesseract.image_to_string(im_bw, lang="eng", config=my_config)
    # print(f"Final Text: ", text, end="\n\n\n")

    dilation = remove_noise(im_bw)
    cv2.imwrite(no_noise_path, dilation)
    # contours, hierarchy = cv2.findContours(
    #     dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    # )
    # print(contours)
    # print(hierarchy)
    # cnts = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    # for i in range(len(cnts)):
    #     x, y, w, h = cv2.boundingRect(cnts[i])
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0))
    # cv2.imwrite("rectangle.jpg", inverse)

    # cv2.imwrite("images/final.jpg", image)

    ero_im_thin = thin_or_thick_font(image)
    # im_thick = thin_or_thick_font(im_bw, thick=True)
    cv2.imwrite("images/erode.jpg", ero_im_thin)

    for i in (im_bw, ero_im_thin, dilation):
        text = pytesseract.image_to_string(i, lang="eng", config=my_config)
        print(f"Final Text {i}: ", text, end="\n\n\n")

    # contours, _ = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # for c in contours:
    #     if len(c) >= 5:
    #         x, y, w, h = cv2.boundingRect(c)
    #         if w > 180 and h > 100:
    #             roi = im_bw[y : y + h, x : x + w]

    #             # Show the ROI for debugging
    #             cv2.imshow("ROI", roi)
    #             cv2.waitKey(0)

    #             text = pytesseract.image_to_string(roi, lang="eng", config=my_config)
    #             print(f"#########\n{text.split()}\n")

    cv2.destroyAllWindows()

    # cv2.imwrite("images/dilate.jpg", im_thick)
    # final_text = pytesseract.image_to_string(im_bw, lang="eng", config=my_config)
    # eroded_text = pytesseract.image_to_string(ero_im_thin, lang="eng", config=my_config)
    # dilated_text = pytesseract.image_to_string(im_thick, lang="eng", config=my_config)
    # dilated_text = pytesseract.image_to_string(dilation, lang="eng", config=my_config)
    # grey_text = pytesseract.image_to_string(grey, lang="eng", config=my_config)
    # print("Eroded Text: ", eroded_text)
    # print("Dilated Text: ", dilated_text)
    # print("Grey Text: ", grey_text)


def remove_noise(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones([2, 2], np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones([1, 1], np.uint8)
    image = cv2.erode(image, kernel, iterations=10)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    image = cv2.bitwise_not(image)
    return image


def adapt_thresh(image):
    # image = cv2.bitwise_not(image)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    image = cv2.bitwise_not(image)
    image = cv2.fastNlMeansDenoising(image, h=60)
    image = cv2.GaussianBlur(image, (7, 7), 0)
    cv2.imwrite("images/blur.jpg", image)
    image = cv2.dilate(image, kernel_dilate, iterations=2)
    image = cv2.erode(image, kernel_erode, iterations=1)
    image = cv2.bitwise_not(image)
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5
    )

    cv2.imwrite("images/threshold.jpg", image)
    return image


def thin_or_thick_font(image, thick: bool = False):
    """thicken or thin text in image

    Args:
        image (MatLike): image object
        thick (bool, optional=False): Determines whether to thicken (dilate) or thin (erode) text

    Returns:
        _type_: _description_
    """
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    # if thick:
    image = cv2.dilate(image, kernel, iterations=1)
    # else:
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image


if __name__ == "__main__":
    process_image()
