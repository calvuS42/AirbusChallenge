"""Define the dataset creation functions.

    This script requires 'os', 'pandas', 'numpy', 'cv2'
    to be installed within the Python environment
    you are running this script in.

    Also, this script requires function from local utils
    that can be found in utils directory within this repository.

    This file should be imported as a module and contains the following
    functions:

    * make_image_gen - Create the generator that returns plain images
        in batch size.
    * create_aug_gen - Create the generator with transformed images.
"""
import os
import pandas as pd
import numpy as np
import cv2

from utils.local_utils import get_image_mask


def make_image_gen(ids: pd.Series,
                   dataset: pd.DataFrame,
                   img_folder_path: os.path,
                   batch_size: int,
                   shuffle: bool = True,
                   seed: int = None,):
    '''Create the generator that returns plain images in batch size.

    Parameters
    ----------
    ids: pd.Series
        pd.Series that contains image names.
    dataset: pd.DataFrame
        pd.DataFrame with masks informations.
    img_folder_path: os.path
        path to directory that contains plain images.
    batch_size: int
        number of images in 1 batch.
    shuffle: bool, optional
        if shuffle the images for the next epoch.
    seed: int, optional
        seed to get reproducable results.

    Returns
    -------
    generator
        generator with images in batch size
    '''
    if seed is not None:
        np.random.seed(seed)

    ids = ids.to_numpy()
    out_rgb = []
    out_mask = []
    while True:
        if shuffle:
            np.random.shuffle(ids)
        for image_name in ids:
            img_path = os.path.join(img_folder_path, image_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = get_image_mask(image_name, dataset)
            mask = np.expand_dims(mask, -1)

            out_rgb.append(img)
            out_mask.append(mask)

            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask = [], []


def create_aug_gen(in_gen, aug_generator, seed: int = None):
    '''Create the generator with transformed images.

    Parameters
    ----------
    in_gen
        input generator with images.
    aug_generator : tensorflow.keras.preprocessing.image.ImageDataGenerator
        configured ImageDataGenerator to transform the images.
    seed: int, optional
        seed to get reproducable results.

    Returns
    -------
    generator
        generator with images in batch size
    '''
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        # keep the seeds syncronized otherwise the augmentation to the images
        # are different from the masks
        seed = np.random.choice(range(9999))

        # apply transformations for the x and y
        g_x = aug_generator.flow(
            255*in_x,
            batch_size=in_x.shape[0],
            seed=seed,
            shuffle=True
        )
        g_y = aug_generator.flow(
            in_y,
            batch_size=in_x.shape[0],
            seed=seed,
            shuffle=True
        )

        yield next(g_x)/255.0, next(g_y)
