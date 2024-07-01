import io
import itertools
import os
import pprint
import pytesseract
import cv2
import numpy as np

from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive"]


def handle_auth():
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=8000)  # 8000
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    fetch_lot_from_gdrive("CO-DAR", credentials=creds)


def fetch_lot_from_gdrive(lot_str: str, credentials):
    with build("drive", "v3", credentials=credentials) as service:
        collections = service.files()
        request = collections.list(
            q=f"name contains '{lot_str}' and mimeType = 'application/pdf'"
        )

        try:
            response = request.execute()
            pprint.pp(response, indent=4)
            # data = json.dumps(response, sort_keys=True, indent=4)
            # print(data)
            if response:
                files = response.get("files", [])
                files_data = []

                for f in files:
                    if f["kind"] == "drive#file":
                        files_data.append(f)

            grouped = itertools.groupby(files_data, key=lambda x: (x["id"], x["name"]))
            groups = [group for group, _ in grouped]
            pprint.pp(groups, indent=4)

        except HttpError as e:
            print(f"Error response : {e}")


def generate_jpg_file(filename: str):
    dirpath = f"{os.getcwd()}/images"

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    abs_image_path = os.path.join(f"{dirpath}/{filename}.jpg")
    if os.path.exists(abs_image_path):
        os.remove(abs_image_path)
        print(f"{filename.split("/")[-1]} deleted")
    return abs_image_path


def convert_pdf_to_image(pdf_path):
    pages = convert_from_path(pdf_path, 300)
    for page in pages:
        abs_image_path = generate_jpg_file("out")
        page.save(abs_image_path, "JPEG")
    return abs_image_path


def enhance_image(image_path: str):
    with Image.open(image_path) as image:
        image = image.convert("L")
        first_img_path = generate_jpg_file("first")
        image.save(first_img_path, "JPEG")

        image = ImageEnhance.Contrast(image).enhance(2.4)
        contrast_path = generate_jpg_file("contrast")
        image.save(contrast_path, "JPEG")
        image = image.filter(ImageFilter.MedianFilter)

        image = image.point(
            lambda x: 0 if x < 128 else 255, "1"
        )  # Convert each pixel to black or white
        filter_debug = generate_jpg_file("filter_debug")
        image.save(filter_debug, "JPEG")

    return image


def scan_with_fixed_box(image_path: str, box_size: tuple, step_size: tuple):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape

    print(f"Image dimensions: width={width}, height={height}")

    boxes = []
    for y in range(0, height, step_size[1]):
        for x in range(0, width, step_size[0]):
            box = image[y : y + box_size[1], x : x + box_size[0]]
            print(box.shape)
            # print(
            #     f"Scanning box at position x={x}, y={y}, width={box_size[0]}, height={box_size[1]}"
            # )
            if (
                box.shape[0] == box_size[1] and box.shape[1] == box_size[0]
            ):  # Ensure box is of correct size
                boxes.append(box)
                # print(f"Box added: top-left corner=({x},{y}), shape={box.shape}")
            else:
                print(f"Skipping box at ({x},{y}), shape={box.shape}")

    print(f"Total boxes scanned: {len(boxes)}")
    return boxes


def extract_text_from_ocr_boxes(boxes):
    extracted_text = []

    for i, box in enumerate(boxes):
        box_image = Image.fromarray(box)

        text = extract_text(box_image)
        extracted_text.append(text)
        box_path = generate_jpg_file(f"box_{i}")
        box_image.save(box_path, "JPEG")
    return extracted_text


def extract_text(image):
    # image = enhance_image("out.jpg")
    # image.save("out2.jpg")
    text = pytesseract.image_to_string(image, config=r"--oem 3 --psm 6")  # --oem 3
    # print("\n".join([line for line in text.splitlines() if len(line)]))
    # print(text)
    return text


def parse(pdf_text):
    fields = {}

    for line in pdf_text:
        if "_" in line:
            fields[line.split("_")[0].strip()] = line.split("_")[1].strip()

    for field in fields:
        print(f"{field}: {fields[field]}")


def parse_text_to_struct(lines):
    expected_fields = {
        "name": "NAME",
        "age": "AGE",
        "sex": "SEX",
        "where_buried": "WHERE BURIED",
        "date_of_burial": "DATE OF BURIAL",
        "cause_of_death": "CAUSE OF DEATH",
        "late_residence": "LATE RESIDENCE",
        "undertaker": "UNDERTAKER",
        "remarks": "REMARKS",
    }
    parsed_data = {key: "" for key in expected_fields.keys()}

    for i, line in enumerate(lines):
        for field, keyword in expected_fields.items():
            if i + 1 < len(lines) and keyword in line.upper():
                if keyword == "AGE":
                    age = line.replace("AGE ", "").replace("NAME ", "")
                    parsed_data[field] = age
                else:
                    parsed_data[field] = lines[i + (1)].strip()

    return parsed_data


def extract_and_parse_text(image_path):
    image_path = convert_pdf_to_image("test.pdf")
    image = enhance_image(image_path)

    boxes = scan_with_fixed_box(
        image_path,
        box_size=(image.width, image.height // 3),
        step_size=(image.width // 2, image.height // 8),
    )
    texts = extract_text_from_ocr_boxes(boxes)

    for i, text in enumerate(texts):
        print(f"Box {i} text: {text}")

    # text = pytesseract.image_to_string(image, config=r"--oem 3 --psm 6")
    text = "\n".join(texts)
    lines = text.split("\n")
    print(str(texts).strip().split("\n"))
    parsed_data = parse_text_to_struct(lines)
    pprint.pp(parsed_data, indent=2)


if __name__ == "__main__":
    extract_and_parse_text("test.pdf")
