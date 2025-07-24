# Dates_BSc_Project
This public repository is a clean, shareable version of the original private research repository
##### Note:
The dataset—comprising fruit images and biological measurements—remains the property of the Faculty of Agriculture and is therefore excluded from this public version

The included Random Forest classifier was trained on this proprietary dataset

### Contributions:

•	Classifier implementation: Elhanan Kadosh & Harel Shpunt (SCE)

•	Image processing & file management: Noam Navon (SCE)

### Usage Instructions

Since the data is excluded, you can still:
1.	Upload your own image and apply the provided image processing pipeline
2.	Classify fruit maturity using the trained Random Forest classifier (main.py)

### Recommended Input Image Setup

To get accurate results:

•	Place a single date fruit in the center of the image

•	Set the fruit on white paper, and place a black 1cm² square in the top-right corner (background.pdf for printing is provided)

•	Use diffused ambient light. Avoid direct lighting, which can cause hard shadows and hurt segmentation quality

The images in the data folder are good examples
