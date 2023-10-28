import numpy as np
import pandas as pd
from PIL import Image
import cv2
import tifffile


def show_high_res_rgb(path):
    img = Image.open(path)
    img.save("HR_VISUAL.png")


def show_low_res_rgb(path):
    image = tifffile.imread(path)
    arr = np.array([image[:, :, 4], image[:, :, 3], image[:, :, 2]])
    img = Image.fromarray((np.transpose(arr, (1,2,0)) * 256).astype('uint8'))
    img.save("LR_VISUAL.png")


show_high_res_rgb("train_split/Amnesty POI-1-1-1/Amnesty POI-1-1-1_rgb.png")
show_low_res_rgb("train_split/Amnesty POI-1-1-1/Amnesty POI-1-1-1-1-L2A_data.tiff")