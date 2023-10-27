# Run this first to create the complete data folder

import os
import shutil
from tqdm import tqdm

project_folder = ".."
hr_folder = "hr_dataset"
lr_l2a_folder = "lr_dataset_l2a"
output_path = "Dataset"

os.mkdir(output_path)

hr = os.path.join(project_folder, hr_folder)
for folder in tqdm(os.listdir(hr)):
    if os.path.isdir(os.path.join(hr, folder)):
        os.mkdir(os.path.join(output_path, folder))
        for file in os.listdir(os.path.join(hr, folder)):
            if "rgb." in file:
                shutil.copy(os.path.join(hr, folder, file), os.path.join(output_path, folder))
l2a = os.path.join(project_folder, lr_l2a_folder)
for folder in tqdm(os.listdir(l2a)):
    if os.path.isdir(os.path.join(l2a, folder)):
        for file in (os.listdir(os.path.join(l2a, folder, "L2A"))):
            if "L2A_data" in file:
                shutil.copy(os.path.join(l2a, folder, "L2A", file), os.path.join(output_path, folder))