
Lab 2 - Feature Engineering and Analysis

Overview
--------
This lab involves working with labeled satellite image data stored in `.npz` files. The goal is to perform feature extraction, compute statistical relationships, and engineer new texture-based features to enhance classification capabilities.

File Structure
--------------
lab2/
├── image_data/                  # Folder containing .npz data files
├── lab2.qmd                    # Quarto report
├── enhanced_features_no_skimage.csv  # Output CSV with engineered features
└── Screenshot_*.png            # Optional images used in the report

Part 1: Data Loading and Feature Analysis
-----------------------------------------
Description:
The script performs the following steps:
- Loads specific `.npz` files (`O013490.npz`, `O013257.npz`) that include labeled data.
- Extracts 11 features including NDAI, SD, CORR, and feat4–feat8.
- Calculates:
  - Mutual Information scores using `sklearn`
  - Random Forest feature importances
  - Correlation matrix (visualized with seaborn)

Key Functions:
- load_npz_data_with_labels()
- plot_correlation_matrix()
- compute_mutual_information()
- compute_random_forest_importance()

Part 2: Texture Feature Engineering
-----------------------------------
Description:
This section creates new texture-based features from NDAI using a rolling window approach:
- Contrast: max - min in the window
- Homogeneity: inverse of variance
- Entropy: based on local histogram distribution

Key Function:
- compute_texture_features(df, window_size=5)

The final enhanced DataFrame is saved to `enhanced_features_no_skimage.csv` for further analysis.

Requirements
------------
- Python 3.x
- Libraries:
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - scipy

Install dependencies via pip:
pip install numpy pandas scikit-learn matplotlib seaborn scipy

Running the Analysis
---------------------
1. Run the Python scripts as standalone files or through a Jupyter notebook.
2. Engineered features will be saved as `enhanced_features_no_skimage.csv`.

Output
------
- Printed mutual information scores and feature importances.
- Correlation heatmap visual.
- CSV file with enhanced features.

