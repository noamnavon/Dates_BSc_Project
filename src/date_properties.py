import cv2
import numpy as np


def fruit_size(segmented_image, pixels_in_cm):
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    mask = (segmented_image > 0).astype(np.uint8)

    horizontal_profile = np.sum(mask, axis=0)
    vertical_profile = np.sum(mask, axis=1)

    fruit_width = np.max(vertical_profile) / pixels_in_cm
    fruit_height = np.max(horizontal_profile)  / pixels_in_cm

    return fruit_width, fruit_height

def fruit_color(segmented_image):
    image_lab = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2Lab)

    mask = np.any(segmented_image != [0, 0, 0], axis=-1)
    masked_lab = image_lab[mask]

    mean_lab = np.mean(masked_lab, axis=0)
    var_lab = np.var(masked_lab, axis=0)
    return mean_lab, var_lab
