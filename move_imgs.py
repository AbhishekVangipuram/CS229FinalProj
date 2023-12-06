import os
import shutil
from PIL import Image
import tifffile
import numpy as np

def lr_to_rgb(path, save=True):
  image = tifffile.imread(path)
  arr = np.array([image[:, :, 4], image[:, :, 3], image[:, :, 2]])
  img = Image.fromarray((np.transpose(arr, (1,2,0)) * 256).astype('uint8'))
  if save:
    img.save(path[:-5] + ".png")
  return img


project_folder = "."
hr_folder = "hr_dataset"
lr_l1c_folder = "lr_dataset_l1c"
lr_l2a_folder = "lr_dataset_l2a"
output_path = ".\\data"

hr = os.path.join(project_folder, hr_folder)
for folder in os.listdir(hr):
    if os.path.isdir(os.path.join(hr, folder)):
        if not os.path.exists(os.path.join(output_path, folder)):
            os.makedirs(os.path.join(output_path, folder))
        for file in os.listdir(os.path.join(hr, folder)):
            if "rgb." in file:
                shutil.copy(os.path.join(hr, folder, file), os.path.join(output_path, folder))
l1c = os.path.join(project_folder, lr_l1c_folder)
for folder in os.listdir(l1c):
    if os.path.isdir(os.path.join(l1c, folder)):
        for file in (os.listdir(os.path.join(l1c, folder, "L1C"))):
            if "L1C_data" in file:
                shutil.copy(os.path.join(l1c, folder, "L1C", file), os.path.join(output_path, folder))
l2a = os.path.join(project_folder, lr_l2a_folder)
for folder in os.listdir(l2a):
    if os.path.isdir(os.path.join(l2a, folder)):
        for file in (os.listdir(os.path.join(l2a, folder, "L2A"))):
            if "L2A_data" in file:
                shutil.copy(os.path.join(l2a, folder, "L2A", file), os.path.join(output_path, folder))


for folder in os.listdir(output_path):
    for file in os.listdir(os.path.join(output_path, folder)):
        if '.tiff' in file:
            lr_to_rgb(os.path.join(output_path, folder, file))