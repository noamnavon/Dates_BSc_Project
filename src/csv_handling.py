import csv
import cv2
from tqdm import tqdm
import os
from images_handling import extract_visual_measures
import pandas as pd
import numpy as np


def make_visuals_csv(root_dir):
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename != ".DS_Store":
                image_paths.append(os.path.join(dirpath, filename))

    # Sort image paths numerically based on image number
    image_paths.sort(key=lambda x: int(os.path.basename(x)[4:-4]))

    output_csv_path = "../data/visual_data.csv"
    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "cv_width", "cv_height", "cv_mean_lab", "cv_var_lab"])

        for path in tqdm(image_paths, desc="Processing images"):
            image = cv2.imread(path)
            width, height, mean_lab, var_lab, _ = extract_visual_measures(image)
            writer.writerow([
                os.path.basename(path),
                round(width, 4),
                round(height, 4),
                [round(x, 4) for x in mean_lab],
                [round(x, 4) for x in var_lab]
            ])

def filter_redundant_repeats():
    input_csv = "../data/visual_data.csv"
    output_csv = "../data/visual_data_filtered.csv"

    with open(input_csv, mode="r", newline="") as infile, open(output_csv, mode="w", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        writer.writerow(header)

        skip_range_start = 1585
        skip_range_end = 1746

        i_in_range = 0

        for row in reader:
            img_name = row[0]
            img_number = int(img_name[4:-4])

            if skip_range_start <= img_number <= skip_range_end:
                if (i_in_range % 4) == 0:
                    writer.writerow(row)
                i_in_range += 1
            else:
                writer.writerow(row)

def separate_columns():
    input_csv = "../data/visual_data_filtered.csv"
    output_csv = "../data/visual_data_separated.csv"

    df = pd.read_csv(input_csv)

    df[['cv_mean_L', 'cv_mean_a', 'cv_mean_b']] = df['cv_mean_lab'].apply(lambda x: pd.Series(eval(x, {"np": np})))
    df[['cv_var_L', 'cv_var_a', 'cv_var_b']] = df['cv_var_lab'].apply(lambda x: pd.Series(eval(x, {"np": np})))
    df.drop(columns=["cv_mean_lab", "cv_var_lab"], inplace=True)
    df.to_csv(output_csv, index=False)


def merge_features():
    input_csv1 = "../data/20241231_All_Data_Wet_For_Raw_Image.csv"
    make_visuals_csv("../data/Converted JPEG2000")
    filter_redundant_repeats()
    separate_columns()
    input_csv2 = "../data/visual_data_separated.csv"
    output_csv = "../data/all_data.csv"

    # Load and reorder input_csv1
    with open(input_csv1, newline="") as f1:
        r1 = csv.reader(f1)
        header1 = next(r1)
        idx_area = header1.index("Area")
        data1 = list(r1)

    # Keep original order of non-E and E rows separately
    non_e_rows = [row for row in data1 if row[idx_area] != "E"]
    e_rows = [row for row in data1 if row[idx_area] == "E"]
    reordered_data1 = non_e_rows + e_rows

    # Load input_csv2
    with open(input_csv2, newline="") as f2:
        r2 = csv.reader(f2)
        header2 = next(r2)
        data2 = list(r2)

    # Merge up to the shorter list
    min_len = min(len(reordered_data1), len(data2))

    with open(output_csv, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header1 + header2[1:])
        for i in range(min_len):
            writer.writerow(reordered_data1[i] + data2[i][1:])