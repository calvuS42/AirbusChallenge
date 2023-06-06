"""Define the utils function that are used in multiple other files.

    This script requires 'os', 'pandas', 'numpy', 'tensorflow', 'cv2',
    'yaml', 'skimage' to be installed within the Python environment
    you are running this script in.

    This file should be imported as a module and contains the following
    functions:

    * rle_decode - Decode RLE format into image mask.
    * rle_encode - Encode image mask into RLE format.
    * multi_rle_encode - Encode image mask into multiple
        masks of individual ships
    * get_image_mask - Extract image mask from DataFrame by image name.
    * get_image - Extract image in RGB format from folder.
    * get_config - Extract config with all needed parameters by it's path.
"""
import os
import pandas as pd
import numpy as np
import cv2
import yaml
from skimage.morphology import label


def rle_decode(mask_rle: str, shape: tuple = (768, 768)):
    '''Decode RLE format into image mask.

    Parameters
    ----------
    mask_rle : str
        run-length as string formated (start length)
    shape : tuple, optional
        (height,width) of array to return

    Returns
    -------
    numpy array
        1 - mask, 0 - background

    Reference
    ---------
    https://www.kaggle.com/paulorzp/run-length-encode-and-decode
    '''
    strings = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (strings[0:][::2],
                                                          strings[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for low, high in zip(starts, ends):
        img[low:high] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def rle_encode(img):
    '''Encode image mask into RLE format.

    Parameters
    ----------
    img : numpy array
        1 - mask, 0 - background

    Returns
    -------
    str
        run length as string formated

    Reference
    ---------
    https://www.kaggle.com/paulorzp/run-length-encode-and-decode
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def multi_rle_encode(img):
    '''Encode image mask into multiple masks of individual ships in RLE format.

    Parameters
    ----------
    img : numpy array
        1 - mask, 0 - background.

    Returns
    -------
    list
        list of strings for masks in RLE format.

    Reference
    ---------
    https://www.kaggle.com/code/kmader/from-trained-u-net-to-submission-part-2/notebook
    '''
    labels = label(img[:, :, 0])
    return [rle_encode(labels == k) for k in np.unique(labels[labels > 0])]


def get_image_mask(img_name: str, df: pd.DataFrame):
    '''Extract image mask from DataFrame by image name.

    Parameters
    ----------
    img_name : str
        name of image without any path variables
    df: pd.DataFrame
        pd.DataFrame with 'ImageId', 'ImageHeight', 'ImageWidth',
        'EncodedPixels' columns that contains masks in RLE format.

    Returns
    -------
    numpy array
        1 - mask, 0 - background.

    In order to properly function, df.ImageId should contain img_name
    and has it's height and width. df.EncodedPixels could be pd.Nan,
    that would be treated as an empty mask for the image.
    '''
    all_masks = df[df.ImageId == img_name]

    mask = np.zeros((all_masks['ImageHeight'].to_numpy()[0].astype(int),
                    all_masks['ImageWidth'].to_numpy()[0].astype(int)),
                    dtype=np.int8)

    if pd.isna(all_masks.EncodedPixels).all():
        return mask

    return all_masks.apply(lambda x: rle_decode(
        x['EncodedPixels'],
        (int(x['ImageHeight']), int(x['ImageWidth']))
        ), axis=1).sum()


def get_image(folder, img_name):
    '''Extract image in RGB format from folder.

    Parameters
    ----------
    folder
        folder that contains image.
    img_name
        name of image without any path variables.

    Returns
    -------
    numpy.array
       RGB image as numpy array.
    '''
    img = cv2.imread(os.path.join(folder, img_name))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_config(config_path: os.path) -> dict:
    '''Extract config with all needed parameters by it's path.

    Parameters
    ----------
    config_path : os path
        path to yaml config file.

    Returns
    -------
    dict
        config as dict object.

    Raises
    ------
    yaml.YAMLError
        if config could not be safe_load
    '''
    with open(config_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
