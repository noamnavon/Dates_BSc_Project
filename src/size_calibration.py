import numpy as np
import cv2
from matplotlib import pyplot as plt


def calibrate_size(image, show_image=False):
    """
    Args:
        image: An RGB standard image imported using opencv
        show_image: Show image on screen
    Returns:
        Side size in pixels per cm of the biggest black square of the reference
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape
    x_start = width * 5 // 6
    y_end = height * 1 // 4
    cropped_image = gray_image[:y_end, x_start:]

    binary = (cropped_image > 50).astype(np.uint8)
    binary_scaled = binary * 255
    binary_scaled = retain_largest_black_cluster(binary_scaled)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary_smoothed = cv2.morphologyEx(binary_scaled, cv2.MORPH_OPEN, kernel)

    num_pixels_per_cm = np.sqrt(np.sum(binary_smoothed == 0))
    if show_image:
        plt.imshow(cropped_image, cmap='gray')
        plt.show()
        plt.imshow(binary, cmap='gray')
        plt.show()
        plt.imshow(binary_scaled, cmap='gray')
        plt.show()
        plt.imshow(binary_smoothed, cmap='gray')
        text = f"Resolution: \n{round(num_pixels_per_cm, 2)} [px/cm]"
        plt.text(10, 20, text, color='black', fontsize=70, va='top')  # adjust (x, y) as needed
        plt.show()
    return num_pixels_per_cm

def retain_largest_black_cluster(img):
    mask = np.uint8(img == 0)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return img
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.where(labels == largest_label, 0, 255).astype(np.uint8)
    return out
