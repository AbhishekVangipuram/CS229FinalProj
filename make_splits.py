# Run this to produce folders of data for each split

import pandas as pd
import os
import shutil
from tqdm import tqdm

# os.mkdir("train_split")
# os.mkdir("val_split")
# os.mkdir("test_split")

split = "stratified split.csv"

df = pd.read_csv(split).drop_duplicates()

for index, row in tqdm(df.iterrows()):
    if not os.path.exists(f"{row['split']}_split/{row['tile']}"):
        shutil.copytree(f"data/{row['tile']}", f"{row['split']}_split/{row['tile']}")