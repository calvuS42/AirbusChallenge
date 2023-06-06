'''Script for training model.

    This script allows to train previously defined in model_utils
    model. While training it saves model checkpoints for easier
    initialization in future. An important part of this script is
    config that contains almost all necessary parameters. So, it's
    more convenient to tune parameters from it.

    This script requires next parameters from command line:
    * config_path - path for a config file with model params.
    * checkpoint_path - directory to save checkpoints after each epoch.

    This script requires 'pandas', 'tensorflow', 'argparse'
    to be installed within the Python environment
    you are running this script in.

    Also, this script requires objects that was implemented in packages
    that can be found in utils directory within this repository.
'''
# libs imports
import argparse
import pandas as pd
from tensorflow import keras as K
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# imports of custom tools
from utils.model_utils import UNET, dice_coef_loss, IoU, plot_history, \
    dice_coef
from utils.dataset_utils import make_image_gen, create_aug_gen
from utils.local_utils import get_config

# set-up the random seed to get reproducible results
SEED = 42


def generate_dataset(config_file):
    '''Create the generator that returns plain images in batch size.

    Parameters
    ----------
    config_file
        config file with required parameters.

    Returns
    -------
    tuple
        two generators with images train and validation sets.
    '''
    # upload all necessary text data with image ids for both
    # train and validation sets. 
    df = pd.read_csv(config_file['path']['dataset_path'])
    train_ids = pd.read_csv(config_file['path']['train_ids'])
    val_ids = pd.read_csv(config_file['path']['val_ids'])

    # create ImageDataGenerator for data augmentations
    train_datagen = ImageDataGenerator(**config_file['train_datagen_params'])

    # initialize the train data generator with plain images
    training_data = make_image_gen(
        ids=train_ids.ImageId,
        dataset=df,
        seed=SEED,
        **config_file['dataset_params']
    )

    # initialize the validation data generator with plain images
    validation_data = make_image_gen(
        ids=val_ids.ImageId,
        dataset=df,
        seed=SEED,
        **config_file['dataset_params']
    )

    # apply augmentations to the train data
    training_data = create_aug_gen(training_data, train_datagen, seed=SEED)

    return training_data, validation_data


if __name__ == '__main__':
    # Get model config path from command line
    parser = argparse.ArgumentParser(prog='train.py',
                                     description=__doc__)
    parser.add_argument('--config_path',
                        default='./config.yaml',
                        help='Enter path to the config file')
    parser.add_argument('--checkpoint_path',
                        default='./weights',
                        help='Enter path to the checkpoint directory')
    args = parser.parse_args()

    # get the config parameters
    config = get_config(args.config_path)

    # get the train and validation data with parameters
    train_data, val_data = generate_dataset(config_file=config)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.checkpoint_path,
        save_weights_only=True,
        verbose=1
    )

    # initialize model
    model = UNET(**config['model_params']).model
    model.summary()

    # setup model for training
    adam = K.optimizers.Adam(
        learning_rate=config['training_params']['learning_rate']
    )
    model.compile(
        optimizer=adam,
        loss=dice_coef_loss,
        metrics=[dice_coef, IoU],
        run_eagerly=True
    )

    # train model
    hist = model.fit(
        train_data,
        epochs=config['training_params']['epochs'],
        steps_per_epoch=config['training_params']['steps_per_epoch'],
        validation_steps=config['training_params']['validation_steps'],
        callbacks=[cp_callback],
        validation_data=val_data
    )

    plot_history(hist, config['path']['checkpoint_path'])
