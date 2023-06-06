"""Creating the csv file in appropriate format for submission on Kaggle.

    This script allows to generate csv file with model
    prediction to submitt to Kaggle competition:
    https://www.kaggle.com/competitions/airbus-ship-detection

    This script requires next parameters from command line:
     * config_path - path for a config file with model params.
     * test_image_dir - directory with images for submission.
     * checkpoint_dir - directory with checkpoints from training.
     * output_path - path to save output file.

    This script requires 'os', 'pandas', 'numpy', 'tensorflow',
    to be installed within the Python environment
    you are running this script in.

    Also, this script requires objects that was implemented in packages
    that can be found in utils directory within this repository.
"""

import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf

from utils.local_utils import multi_rle_encode, get_config
from utils.model_utils import UNET, predict
from utils.local_utils import get_image


if __name__ == '__main__':

    # Get all necessary parameters from command line
    parser = argparse.ArgumentParser(prog='create_submission_file.py',
                                     description=__doc__)
    parser.add_argument('--config_path',
                        default='./config.yaml',
                        help='Enter path to the config file')
    parser.add_argument('--test_image_dir',
                        default='./data/test_v2/',
                        help='Enter path to the directory with images \
                             you want to create masks for')
    parser.add_argument('--checkpoint_dir',
                        default='./weights',
                        help='Enter path to the checkpoint directory \
                              for model')
    parser.add_argument('--output_path',
                        default='./',
                        help='Enter path where to save the submission.csv')

    args = parser.parse_args()

    # get config with model params that are used for model initialization
    config = get_config(args.config_path)

    # initialize model and load weights of the latest checkpoint
    # from checkpoint dir
    model = UNET(config['model_params']['num_classes']).model
    latest = tf.train.latest_checkpoint(args.checkpoint_dir)
    model.load_weights(latest)

    # get image names from test directory
    test_image_dir = args.test_image_dir
    test_paths = os.listdir(test_image_dir)

    # entering main loop
    out_pred_rows = []
    for i, image_name in enumerate(test_paths):

        # print progress information into console
        if not i % 1000:
            print(f'Processing {i}th image!')

        # get rgb image by it's name
        img = get_image(test_image_dir, image_name)
        img = img/255.0

        # predict mask for image via model
        cur_seg = predict(img, model)
        cur_seg = np.expand_dims(cur_seg, -1)

        # split mask into individual ships masks and
        # store it in the list
        cur_rles = multi_rle_encode(cur_seg)
        if len(cur_rles) > 0:
            for c_rle in cur_rles:
                out_pred_rows += [{'ImageId': image_name,
                                   'EncodedPixels': c_rle}]
        else:
            out_pred_rows += [{'ImageId': image_name, 'EncodedPixels': None}]

    # create DataFrame in proper format and save it to output path
    submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
    submission_df.to_csv(os.path.join(args.output_path, 'submission.csv'),
                         index=False)
