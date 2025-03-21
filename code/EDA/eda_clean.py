
import os
import pandas as pd
import numpy as np


def clean(arr):
    """
    Converts a NumPy array into a pandas DataFrame with predefined column names.

    Parameters:
    arr (numpy.ndarray): A 2D NumPy array where each row represents an observation, 
                         and columns correspond to specific features.

    Returns:
    pandas.DataFrame: A DataFrame with the following columns:
        - "y": y-coordinate
        - "x": x-coordinate
        - "NDAI": constructed feature 3
        - "SD": constructed feature 2
        - "CORR": constructed feature 1
        - "raDF", "raCF", "raBF", "raAF", "raAN": angle radiances (ra) by camera (Df, Cf, Bf, Af, An)
        - "label": expert label (+1 = Cloud, 0 = Unlabeled, -1 = Noncloud)
    """

    # convert numpy array to dataframe
    df = pd.DataFrame(
        data=arr,
        columns=["y", "x", "NDAI", "SD", "CORR", "raDF", "raCF", "raBF", "raAF", "raAN", "label"],
    )
    
    return df


if __name__ == "__main__":

    # define directory structure
    home_dir = os.path.expanduser("~")
    data_dir = os.path.join(home_dir, "stat214", "data")

    # define image and file names
    images = ["O012791", "O013257", "O013490"]
    npz_files = [f"{im}.npz" for im in images]
    csv_files = [f"{im}.csv" for im in images]
    
    # load data
    arr1 = np.load(os.path.join(data_dir, npz_files[0]))["arr_0"]
    arr2 = np.load(os.path.join(data_dir, npz_files[1]))["arr_0"]
    arr3 = np.load(os.path.join(data_dir, npz_files[2]))["arr_0"]

    # clean data
    df1 = clean(arr1)
    df2 = clean(arr2)
    df3 = clean(arr3)

    # save data
    df1.to_csv(os.path.join(data_dir, csv_files[0]), index=False)
    df2.to_csv(os.path.join(data_dir, csv_files[1]), index=False)
    df3.to_csv(os.path.join(data_dir, csv_files[2]), index=False)

