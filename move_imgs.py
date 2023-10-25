import os
import shutil

project_folder = ".."
hr_folder = "hr_dataset"
lr_l1c_folder = "lr_dataset_l1c"
lr_l2a_folder = "lr_dataset_l2a"
output_path = "Data"

hr = os.path.join(project_folder, hr_folder)
for folder in os.listdir(hr):
    if os.path.isdir(os.path.join(hr, folder)):
        os.mkdir(os.path.join(output_path, folder))
        for file in os.listdir(os.path.join(hr, folder)):
            if "rgb." in file:
                shutil.copy(os.path.join(hr, folder, file), os.path.join(output_path, folder))
lr1 = os.path.join(project_folder, lr_l1c_folder)
for folder in os.listdir(lr1):
    if os.path.isdir(os.path.join(lr1, folder)):
        for file in (os.listdir(os.path.join(lr1, folder, "L1C"))):
            if "L1C_data" in file:
                shutil.copy(os.path.join(lr1, folder, "L1C", file), os.path.join(output_path, folder))
l2a = os.path.join(project_folder, lr_l2a_folder)
for folder in os.listdir(l2a):
    if os.path.isdir(os.path.join(l2a, folder)):
        for file in (os.listdir(os.path.join(l2a, folder, "L2A"))):
            if "L2A_data" in file:
                shutil.copy(os.path.join(l2a, folder, "L2A", file), os.path.join(output_path, folder))