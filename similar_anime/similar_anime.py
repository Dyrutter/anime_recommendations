import argparse
import logging
# import os
import wandb
from distutils.util import strtobool
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import tensorflow.keras.layers as tfkl
import tensorflow.keras.callbacks as tfkc
import matplotlib.pyplot as plt


# model = tf.keras.models.load_model('./anime_nn.h5')
# weight = model.load_weights('./anime_weights.h5')

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def get_df():
    """
    Get data frame from wandb
    """
    run = wandb.init(project=args.project_name)
    logger.info("Downloading data artifact")
    artifact = run.use_artifact(args.input_data, type='preprocessed_data')
    artifact_path = artifact.file()
    df = pd.read_parquet(artifact_path)

    # Encoding categorical data
    # Get all user ids and anime ids
    user_ids = df["user_id"].unique().tolist()
    anime_ids = df["anime_id"].unique().tolist()

    # Dict of format {user_id: count_number}
    reverse_encoded_user_ids = {x: i for i, x in enumerate(user_ids)}
    reverse_encoded_anime = {x: i for i, x in enumerate(anime_ids)}

    # Dict of format  {count_number: user_id}
    # encoded_user_ids = {i: x for i, x in enumerate(user_ids)}
    # encoded_anime = {i: x for i, x in enumerate(anime_ids)}

    # Convert values of format id to count_number
    df["user"] = df["user_id"].map(reverse_encoded_user_ids)
    df["anime"] = df["anime_id"].map(reverse_encoded_anime)

    # Get number of users and number of anime
    n_users = len(reverse_encoded_user_ids)
    n_animes = len(reverse_encoded_anime)

    # Shuffle, because currently sorted according to user ID
    df = df.sample(frac=1, random_state=42)
    to_drop = ['watching_status', 'watched_episodes', 'max_eps', 'half_eps']
    df = df.drop(to_drop, axis=1)

    # Holdout test set is approximately 10% of data set
    # test_df = rating_df[38000000:]
    # train_df = rating_df[:38000000]

    return df, n_users, n_animes







if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an anime recommendation neural network",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
    	"--main_data",
    	type=str,
    	help="Wandb artifact of main data frame",
    	required=True
    )

    parser.add_argument(
    	"--synopses",
    	type=str,
    	help="Wandb artifact of anime synopses",
    	required=True
    )

    parser.add_argument(
    	"--all_anime",
    	type=str,
    	help="Wandb artifact containing list of all anime",
    	requried=True
    )

    parser.add_argument(
    	"--anime_weights",
    	type=str,
    	help="Wandb artifact containing numpy array of anime weights",
    	required=True
    )

    parser.add_argument(
    	"--user_weights",
    	type=str,
    	help="Wandb artifact containing numpy array of user weights",
    	required=True
    )

    parser.add_argument(
    	"--weights",
    	type=str,
    	help="Wandb artifact with .h5 file of all neural network weights",
    	required=True
    )

    parser.add_argument(
    	"--history",
    	type=str,
    	help="Wandb artifact with .csv file of neural network run history",
    	required=True
    )

    parser.add_argument(
    	"--model",
    	type=str,
    	help="Wandb artifact with .h5 file of neural network",
    	required=True
    )
