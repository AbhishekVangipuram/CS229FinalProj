# similar to multinomial, we flattenn the images
# and use a fully connected ned to categorize the images
# for low and high res images

import util
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

import tensorflow as tf
from tensorflow.keras import layers, models


