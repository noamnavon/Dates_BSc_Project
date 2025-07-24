import cv2
import numpy as np
import os
from paths_handling import collect_image_paths


def get_image_ref_hsv(image):
    height, width, channels = image.shape
    x_end = width * 1 // 5
    y_end = height * 1 // 4
    cropped_image = image[:y_end, :x_end]

    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    values = hsv_image
    mean = np.mean(values, (0,1))
    return mean

def get_global_ref_hsv(image_paths, cache_path):
    if os.path.exists(cache_path):
        return np.load(cache_path)
    else:
        all_means = []
        for path in image_paths:
            image = cv2.imread(path)
            mean = get_image_ref_hsv(image)
            all_means.append(mean)

        global_mean = np.mean(np.array(all_means), axis=0)
        np.save(cache_path, global_mean)
        return global_mean

def calibrate_color_hsv(image):
    cache_path = "../data/ref_mean.npy"
    image_paths = collect_image_paths("../data/Converted JPEG2000")
    global_ref_hsv = get_global_ref_hsv(image_paths, cache_path)
    image_ref_hsv = get_image_ref_hsv(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (global_ref_hsv[1] / image_ref_hsv[1]), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (global_ref_hsv[2] / image_ref_hsv[2]), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)