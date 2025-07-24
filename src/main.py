import cv2
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from images_handling import display_images_mosaic, extract_visual_measures
from csv_handling import merge_features
from classifier import train


_old_imshow = plt.imshow
def _imshow_true_size(img, **kwargs):
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    figsize = (width / dpi, height / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # fills entire figure
    ax.imshow(img, **kwargs)
    ax.axis('off')
    return ax

plt.imshow = _imshow_true_size

def classify(path):
    image = cv2.imread(path)
    width, height, mean, _, _ = extract_visual_measures(image)
    print(width, height, mean)
    cv_features = width, height, mean[0], mean[1]

    model, scaler = joblib.load("../models/random_forest_model.pkl")
    feature_names = ["cv_width", "cv_height", "cv_mean_L", "cv_mean_a"]
    df_features = pd.DataFrame([cv_features], columns=feature_names)
    features_scaled = scaler.transform(df_features)
    prediction = model.predict(features_scaled)
    print("Prediction:", prediction[0])
    return prediction[0]

def graph_cv():
    from itertools import cycle
    df = pd.read_csv("../data/all_data.csv")
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    features = [
        "cv_width", "cv_height",
        "cv_mean_L", "cv_mean_a", "cv_mean_b",
    ]

    fig, axs = plt.subplots(len(features), 1, figsize=(10, 2.5 * len(features)), sharex=True)

    for ax, col in zip(axs, features):
        sns.boxplot(data=df, x="WPP", y=col, ax=ax, color=next(colors))
        ax.set(title=col, ylabel="Value")
        ax.grid(True)

    axs[-1].set_xlabel("Weeks Post Pollination (WPP)")
    plt.tight_layout()
    plt.show()


def compare_size():
    df = pd.read_csv("../data/all_data.csv")
    df["Width_(mm)"] /= 10  # Convert mm to cm
    df["Length_(mm)"] /= 10

    for cv_col, true_col, label in [("cv_width", "Width_(mm)", "Width"), ("cv_height", "Length_(mm)", "Height")]:
        bias = (df[cv_col] - df[true_col]).mean()
        print(bias)
        corrected_col = f"{cv_col}_corrected"
        df[corrected_col] = df[cv_col] - bias

        errors = df[corrected_col] - df[true_col]
        nrmse = ((errors ** 2).mean() ** 0.5) / df[true_col].mean()
        mape = 100 * (errors / df[true_col]).abs().mean()

        plt.scatter(df[corrected_col], df[true_col])
        plt.plot([2, 7], [2, 7], linestyle='--', color='gray', label='y = x')
        plt.xlim(2, 7)
        plt.ylim(2, 7)
        plt.title(f"{label} Comparison", fontsize=14)
        plt.xlabel(f"Predicted {label} (cm)", fontsize=12)
        plt.ylabel(f"True {label} (cm)", fontsize=12)
        plt.text(2.1, 6.3,
                 f"nRMSE: {nrmse:.3f}\nMAPE: {mape:.1f}%",
                 fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
        plt.show()

def color_correlation():
    df = pd.read_csv("../data/all_data.csv")

    color_cols = ["cv_width","cv_height","cv_mean_L","cv_mean_a"]
    targets = [
        "WPP",
        "Length_(mm)",
        "Width_(mm)",
        "L:W_Ratio",
        "Fresh_Weigth_Whole_Fruit_(gr)",
        "Fresh_Weight_Half_Pericarp_(gr)",
        "Fresh_Weight_Seed_(gr)",
        "Dry_Weight_Half_Pericarp_(gr)",
        "Dry_Weigth_Seed_(gr)",
        "Cal._Water_Content_Pericarp_(%)",
        "Cal._Water_Content_Seed_(%)",
        "Cal._Fresh_Weigth_Pericarp_(gr)",
        "Cal._Dry_Weigth_Pericarp_(gr)",
        "TSS_For_Spectral_Analysis_(%)",
        "Suitable_For_Treatment",
    ]

    corr_matrix = df[targets + color_cols].corr().loc[targets, color_cols]

    plt.figure(figsize=(6.5, 8))
    ax = sns.heatmap(corr_matrix,
                     xticklabels=['Width', 'Height', 'L', 'a'],
                     annot=True,
                     fmt=".2f",
                     cmap="coolwarm",
                     center=0, vmin=-1, vmax=1, annot_kws={"size": 11})
    plt.title("Correlation")
    plt.ylabel("Targets")
    plt.xlabel("CV parameters")
    plt.tight_layout()
    plt.show()

    # Add heatmap of color mean inter-correlations
    color_corr = df[color_cols].corr()

    plt.figure(figsize=(4, 3.5))
    sns.heatmap(color_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, annot_kws={"size": 11})
    plt.title("Correlation Between Color Channels")
    plt.ylabel("Color Mean")
    plt.xlabel("Color Mean")
    plt.tight_layout()
    plt.show()

def demo(path):
    image = cv2.imread(path)
    from images_handling import extract_visual_measures
    width, height, mean_lab, var_lab, segmented = extract_visual_measures(image)
    prediction = classify(path)
    if prediction == 0:
        prediction = "too early"
    elif prediction == 1:
        prediction = "just in time"
    elif prediction == 2:
        prediction = "too late"
    plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    text = (f"Width: {round(width, 2)} [cm]\nHeight: {round(height, 2)} [cm]\nMean LAB:"
            + np.array2string(np.round(mean_lab, 2))
            + "\nVar LAB: "
            + np.array2string(np.round(var_lab, 2))
            + f"\nClass: {prediction}")
    plt.text(10, 20, text, color='white', fontsize=100, va='top')  # adjust (x, y) as needed
    plt.show()

def main():
# merge_features()
#    graph_cv()
#    compare_size()
    color_correlation()
#    train()
#    classify("../data/Converted JPEG2000/20230718_18WPP/IMG_1381.jp2")


if __name__ == "__main__":
    main()
