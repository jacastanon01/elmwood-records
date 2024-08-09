import cv2

from handle_image import (
    convert_pdf_to_image,
    read_coordinates,
    select_files,
    crop_segment,
    preprocess_segment,
    extract_text,
)


def main():
    files = select_files("Cards/CO-DAR")
    images = [convert_pdf_to_image(file) for file in files[:5]]

    image = images[0]
    cv2.imwrite("image.jpg", image[0])
    points = read_coordinates("ref_points.txt")

    segments = [
        crop_segment(img[0], start, end) for (start, end) in points for img in images
    ]
    process = [preprocess_segment(segment, i) for i, segment in enumerate(segments)]
    text = [extract_text(segment) for segment in process]
    print(text)


if __name__ == "__main__":
    main()
