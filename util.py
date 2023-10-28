# Utility function to be called for models

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import tifffile

# Pull labels
def get_labels():
    df_labels = pd.read_csv("WorldStrat Dataset.csv", index_col=0)[["SMOD Class"]]
    df_labels = df_labels[~df_labels.index.duplicated(keep='first')]
    df_splits = pd.read_csv("stratified split.csv", index_col="tile")[["split"]]
    df_splits = df_splits[~df_splits.index.duplicated(keep='first')]
    df_unified = pd.merge(df_labels, df_splits, left_index=True, right_index=True)
    df_unified["SMOD Class"] = df_unified["SMOD Class"].map({"Water":0, "Rural: Very Low Dens":1, "Rural: Low Dens":2, "Rural: cluster":3, "Urban: Suburban":4, "Urban: Semi-dense":5, "Urban: Dense":6, "Urban: Centre":7})
    df_final = df_unified.drop("ASMSpotter-1-1-1")
    return df_final


# Produce array of images
def get_high_res():
    labels = get_labels()
    high_res = []
    for index, row in tqdm(labels.iterrows()):
        image = cv2.cvtColor(cv2.imread(f"{row['split']}_split/{index}/{index}_rgb.png"), cv2.COLOR_BGR2RGB)
        high_res.append(image[277:777, 277:777])
    return np.array(high_res)

# Save array to save time
def save_high_res():
    high_res = get_high_res()
    np.save("high_res.npy", high_res)

def get_low_res():
    labels = get_labels()
    low_res = []
    for index, row in tqdm(labels.iterrows()):
        image = tifffile.imread(f"{row['split']}_split/{index}/{index}-1-L2A_data.tiff")
        arr = np.array([image[:, :, 4], image[:, :, 3], image[:, :, 2]])
        img = (np.transpose(arr, (1,2,0)) * 256).astype('uint8')
        low_res.append(img[:147, :147, :])
    return np.array(low_res)


# Give labels and splits for training
def get_labels_and_split():
    labels = get_labels()

    train = labels["split"] == "train"
    val =   labels["split"] == "val"
    test =  labels["split"] == "test"

    y = labels.reset_index(drop=True)["SMOD Class"].to_numpy()

    return y, train, val, test