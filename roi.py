import cv2
import pprint
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import MouseButton
from typing import List, Tuple, Optional


def load_image(image_path: str):
    """
    Load an image from a given path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        Optional[cv2.Mat]: The loaded image or None if loading fails.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        return image
    except Exception as e:
        print(f"Error: {e}")
        return None


def display_image_with_selector(image, onselect_callback, num_selectors=1) -> None:
    """
    Display an image and set up a rectangle selector for region selection.

    Args:
        image (cv2.Mat): The image to display.
        onselect_callback: The callback function for rectangle selection events.
        num_selectors (int): How many rectangles to draw
    """
    fig, ax = plt.subplots()
    ax.imshow(image)

    rect_selector = RectangleSelector(
        ax,
        onselect_callback,
        minspanx=5,
        minspany=5,
        useblit=True,
        button=MouseButton.LEFT,
        spancoords="pixels",
        interactive=True,
        # ignore_event_outside=True,
    )

    plt.show()


def onselect(
    eclick,
    erelease,
    ref_points: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    image,
) -> None:
    """
    Handle the event when a rectangle selection is made.

    Args:
        eclick: Mouse click event.
        erelease: Mouse release event.
        ref_points (List[Tuple[Tuple[float, float], Tuple[float, float]]]): List to store selected rectangle coordinates.
        image (cv2.Mat): The image being displayed.
    """
    rect_coords = (
        (float(eclick.xdata), float(eclick.ydata)),
        (float(erelease.xdata), float(erelease.ydata)),
    )
    ref_points.append(rect_coords)

    plt.gca().clear()
    plt.imshow(image)

    for start, end in ref_points:
        plt.gca().add_patch(
            Rectangle(start, end[0] - start[0], end[1] - start[1], fill=False)
        )

    plt.draw()


def save_ref_points(
    ref_points: List[Tuple[Tuple[float, float], Tuple[float, float]]], json_path: str
) -> None:
    """
    Save the reference points to a JSON file in key-value pair format.

    Args:
        ref_points (List[Tuple[Tuple[float, float], Tuple[float, float]]]): The list of rectangle coordinates.
        json_path (str): The path to the JSON file.
    """
    formatted_points = [
        # {
        #     "start": [float(coords) for coords in start],
        #     "end": [float(coords) for coords in end],
        # }
        (start, end)
        for start, end in ref_points
    ]

    with open(json_path, "w") as f:
        for start, end in ref_points:
            f.write(f"{start[0]} {start[1]} {end[0]} {end[1]}\n")
        # json.dump({"ref_points": formatted_points}, f, indent=4)

    print(f"Reference points saved to {json_path}")


def main(image_path: str, json_path: str) -> None:
    """
    Main function to load an image, display it with a rectangle selector, and save the selected regions.

    Args:
        image_path (str): The path to the image file.
        json_path (str): The path to save the reference points JSON file.
    """
    image = load_image(image_path)
    if image is None:
        return

    ref_points: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    def callback(eclick, erelease):
        onselect(eclick, erelease, ref_points, image)

    display_image_with_selector(image, callback)
    save_ref_points(ref_points, json_path)
    pprint.pp(ref_points, indent=4)


if __name__ == "__main__":
    main("images/blur.jpg", "ref_points.txt")
