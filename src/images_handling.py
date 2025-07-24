import numpy as np
from paths_handling import collect_image_paths
import cv2
import color_calibration_table
import size_calibration
import segmentation
import date_properties
from matplotlib import pyplot as plt


def extract_visual_measures(image, show_image=False):
    calibrated = color_calibration_table.calibrate_color(image, show_image)
    px_per_cm = size_calibration.calibrate_size(image, show_image)
    segmented = segmentation.segment_date(calibrated, show_image)
    width, height = date_properties.fruit_size(segmented, px_per_cm)
    mean_lab, var_lab = date_properties.fruit_color(segmented)
    if show_image:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        plt.imshow(cv2.cvtColor(calibrated, cv2.COLOR_BGR2RGB))
        plt.show()
        plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
        plt.show()
    return width, height, mean_lab, var_lab, segmented

def process_image(image, calibrate, segment):
    if calibrate:
        image = color_calibration_table.calibrate_color(image)
    if segment:
        image = segmentation.segment_date(image)
    return image

def display_images_mosaic(root_folder, skip, resize_scale, calibrate, segment, max_per_row=8):
    image_paths = collect_image_paths(root_folder)
    smalls = []
    for i, path in enumerate(image_paths):
        if i % skip != 0:
            continue
        image = cv2.imread(path)
        image = process_image(image, calibrate=calibrate, segment=segment)
        small = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_LINEAR)
        smalls.append(small)
        print(f"Making Mosaic: {100*i/len(image_paths)}%")
    tiles = []
    row = []
    count = 0
    for small in smalls:
        row.append(small)
        count += 1
        if count % max_per_row == 0:
            min_height = min(img.shape[0] for img in row)
            min_width = min(img.shape[1] for img in row)
            while len(row) < max_per_row:
                row.append(np.zeros((min_height, min_width, 3), dtype=np.uint8))
            row = [img[:min_height, :min_width] for img in row]
            tiles.append(np.hstack(row))
            row = []
    if row:
        min_height = min(img.shape[0] for img in row)
        min_width = min(img.shape[1] for img in row)
        while len(row) < max_per_row:
            row.append(np.zeros((min_height, min_width, 3), dtype=np.uint8))
        row = [img[:min_height, :min_width] for img in row]
        tiles.append(np.hstack(row))

    min_row_width = min(tile.shape[1] for tile in tiles)
    tiles = [tile[:, :min_row_width] for tile in tiles]

    superimage = np.vstack(tiles)
    cv2.imshow('Original', superimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()