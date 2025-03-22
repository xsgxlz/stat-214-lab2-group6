### Exploratory Data Analysis

##### Contents
This directory contains two primary scripts:

1. `eda_clean.py`: This Python script processes the three raw `.npz` files containing the expertly labeled data. It loads the data, assigns appropriate column names and datatypes, and saves each dataset as a `.csv` file in the `data` directory.
   
2. `eda_plots.R`: This R script loads the cleaned `.csv` files into dataframes and generates various exploratory data analysis plots. The resulting figures are saved to the `figs` directory.

##### Dependencies

- Python: 
  - `numpy`
  - `pandas`
  - `os`
  - (See `environment.yaml` for full details)

- R: 
  - `tidyverse`
  - `ggally`
  - `rlang`
  - `glue`
  - `ggtext`
  - `patchwork`
  - (See `environment-r.yaml` for full details)
