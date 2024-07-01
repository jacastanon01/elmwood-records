import os
import pprint
import pytesseract
import cv2

from pdf2image import convert_from_path
from pytesseract import Output
from PIL import Image, ImageEnhance, ImageFilter


def generate_jpg_file(filename: str, image) -> str:
    """creates images folder if one doesn't exist and generates jpg file

    Args:
        filename (str): Name of jpg file
        image (Image): image to save

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


def convert_pdf_to_image(pdf_path: str) -> str:
    pages = convert_from_path(pdf_path, 300)
    for page in pages:
        abs_image_path = generate_jpg_file("pdf_img", page)
    return abs_image_path


def enhance_image(image_path: str):
    with Image.open(image_path) as image:
        image = image.convert("L")
        generate_jpg_file("first", image)

        image = ImageEnhance.Contrast(image).enhance(2.4)
        generate_jpg_file("contrast", image)
        image = image.filter(ImageFilter.MedianFilter)

        image = image.point(
            lambda x: 0 if x < 128 else 255, "1"
        )  # Convert each pixel to black or white

    return image


def parse_text_to_struct(lines):
    expected_fields = {
        "name": "NAME",
        "age": "AGE",
        "sex": "SEX",
        "buried_info": "WHERE BURIED | DATE OF BURIAL",
        "cause_of_death": "CAUSE OF DEATH",
        "late_residence": "LATE RESIDENCE",
        "undertaker": "UNDERTAKER",
        "remarks": "REMARKS",
    }
    parsed_data = {key: "" for key in expected_fields.keys()}
    print(lines)
    i = 0
    while i < len(lines):
        # print(lines[i], end="\n======================\n")
        for field, label in expected_fields.items():

            line = lines[i].strip()
            if line and (line in label or label in line):
                print(line, end="!!!!!!!!\n")
                i += 1
                if lines[i].strip() in label or label in lines[i].strip():
                    i += 1
                value = []
                # print(line, lines[i], field, label, sep=" : ", end="\n-------------\n")

                while (
                    i < len(lines)
                    and lines[i].strip() != ""
                    # and not (lines[i].strip() in label or label in lines[i].strip())
                    # and not any(kw in lines[i] for kw in expected_fields.values())
                ):
                    print(lines[i].strip(), label, sep=" in ")

                    value.append(lines[i].strip())

                    i += 1

                value_str = " ".join(value)
                # print(value)
                parsed_data[field] = value_str
        i += 1

    return parsed_data


def extract_and_parse_text(pdf_path):
    my_config = r"--oem 3 --psm 3"
    pages = convert_from_path(pdf_path, 300)
    image_path = generate_jpg_file("pdf_img", pages[1])

    image = cv2.imread(image_path)
    data = pytesseract.image_to_data(
        image, lang="eng", config=my_config, output_type=Output.DICT
    )
    n_boxes = len(data["level"])

    for i in range(n_boxes):
        (x, y, w, h) = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    found_text = data["text"]

    if found_text is not None:
        parsed_text = parse_text_to_struct(found_text)
    pprint.pp(parsed_text, indent=2)
    # cv2.imshow("img", image)
    # cv2.waitKey(0)


if __name__ == "__main__":
    extract_and_parse_text("test.pdf")
