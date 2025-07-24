import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from paths_handling import collect_image_paths


def get_image_ref(image):
    height, width, channels = image.shape
    x = width * 1 // 5
    y_start = height * 5//6
    cropped_image = image[y_start:, width//2-x:width//2+x]
    mean = np.mean(cropped_image, axis=(0, 1))
    return mean

def get_global_ref(image_paths, cache_path):
    if os.path.exists(cache_path):
        return np.load(cache_path)
    else:
        all_means = []
        for path in image_paths:
            image = cv2.imread(path)
            mean = get_image_ref(image)
            all_means.append(mean)

        global_mean = np.mean(np.array(all_means), axis=0)
        np.save(cache_path, global_mean)
        return global_mean

def calibrate_color(image, show_image=False):
    cache_path = "../data/grey_ref_mean_bgr.npy"
    image_paths = collect_image_paths("../data/Converted JPEG2000")
    global_ref = get_global_ref(image_paths, cache_path).astype(np.float32)
    image_ref_bgr = get_image_ref(image)
    image = image.astype(np.float32)
    correction = global_ref / image_ref_bgr
    if show_image:
        height, width = image.shape[:2]
        x = width // 5
        y1 = height * 5 // 6
        x1, x2 = width // 2 - x, width // 2 + x
        cropped = np.ones_like(image) * 255
        cropped[y1:, x1:x2] = image[y1:, x1:x2]
        plt.imshow(cropped/255)
        text = "Correction factor: " + np.array2string(np.round(correction, 2)) + "\n"
        plt.text(10, 20, text, color='black', fontsize=150, va='top')  # smaller font size to make it visible
        plt.axis('off')
        plt.show()
    image *= correction
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image