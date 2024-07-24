import cv2
import pprint

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import MouseButton


class SelectROI:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.ref_point = []

    def onselect(self, eclick, erelease):
        rect_coords = (
            (eclick.xdata, eclick.ydata),
            (erelease.xdata, erelease.ydata),
        )
        self.ref_point.append(rect_coords)

        plt.gca().clear()

        if hasattr(self, "image_data"):
            plt.imshow(self.image_data)

        for start, end in self.ref_point:
            plt.gca().add_patch(
                Rectangle(start, end[0] - start[0], end[1] - start[1], fill=False)
            )

        plt.draw()

    def save_image(self, save_path: str) -> None:
        plt.savefig(save_path)
        print(f"Image saved to {save_path}")

    def load_image(self):
        try:
            # Load the image
            image = cv2.imread(self.image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at {self.image_path}")
            # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.image_data = image
            return image

        except Exception as e:
            print(f"Error: {e}")
            return None

    def run(self, save_path: str) -> None:
        """
        Run the application: load the image, display it, and set up the RectangleSelector.
        """
        image_gray = self.load_image()
        if image_gray is None:
            return

        _, ax = plt.subplots()
        ax.imshow(image_gray)

        # # Create a rectangle selector
        rect_selector = RectangleSelector(
            ax,
            self.onselect,
            minspanx=5,  # Minimum span in pixels for the rectangle to be valid
            minspany=5,  # Minimum span in pixels for the rectangle to be valid
            useblit=True,  # Use blitting for faster redrawing
            button=MouseButton,  # Button(s) used for selection, 1 for left mouse button
            spancoords="pixels",  # Coordinates are in pixels
            interactive=True,  # Allow interactive updates of the rectangle
            ignore_event_outside=True,  # Allow selection even if dragging starts outside the Axes
        )
        plt.show()

        # self.save_image(save_path)
        pprint.pp(self.ref_point, indent=4)


if __name__ == "__main__":
    roi = SelectROI("images/blur.jpg")
    roi.run("images/plt/roi.png")
