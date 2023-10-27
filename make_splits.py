import pandas as pd
import os
import shutil
from tqdm import tqdm

# os.mkdir("train_split")
# os.mkdir("val_split")
# os.mkdir("test_split")

split = "stratified split.csv"

df = pd.read_csv(split).drop_duplicates()

city_only = df[df["IPCC Class"] == "Settlement"]

for index, row in tqdm(city_only.iterrows()):
    if not os.path.exists(f"{row['split']}_split/{row['tile']}"):
        shutil.copytree(f"Dataset/{row['tile']}", f"{row['split']}_split/{row['tile']}")