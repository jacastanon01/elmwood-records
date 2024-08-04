import cv2

from handle_image import convert_pdf_to_image, select_files


def main():
    files = select_files("Cards/CO-DAR")
    images = [convert_pdf_to_image(file) for file in files]
    image = images[0]
    cv2.imwrite("image.jpg", image[0])
    print(image[1])

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
