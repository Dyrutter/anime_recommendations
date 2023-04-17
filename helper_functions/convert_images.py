import tensorflow as tf
from pillow import Image
import tqdm
import numpy as np
import pandas as pd


def load_image(files, img_shape):
    """
    Load image data into RAM and convert to numpy array
    Scale RGB images into specified shape
    """
    # Create an empty array for images
    cnt = len(files)
    x = np.zeros((cnt,) + img_shape + (3,), dtype=np.float32)
    i = 0
    # Read images in
    for file in tqdm.tqdm(files):
        img = Image.open(file)
        img = img.resize(img_shape)
        img = np.array(img)
        img = img / 255
        x[i, :, :, :] = img
        i += 1
    return x


def convert(dataset, img_shape):
    """
    Prepare data for TPU in tf records
    dataset: String path to dataset
    img_shape: shape of image in tuple form (X, Y)

    """
    df = pd.read_csv(dataset)
    # Add filename column using image ids
    df['filename'] = "img-" + df.id.astype(str) + ".jpg"
    images = [x for x in df.filename]

    # Load images
    x = load_image(images, img_shape)
    y = df.img_num.values

    # Convert to data set
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    return dataset
