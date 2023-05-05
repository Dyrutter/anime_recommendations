import argparse
import logging
# import os
import wandb
import string
import pandas as pd
import numpy as np
import tensorflow as tf
from distutils.util import strtobool



logging.basicConfig(
    filename='./similar_anime.log',  # Path to log file
    level=logging.INFO,  # Log info, warnings, errors, and critical errors
    filemode='a',  # Create log file if one doesn't already exist and add
    format='%(asctime)s-%(name)s - %(levelname)s - %(message)s',
    datefmt='%d %b %Y %H:%M:%S %Z',  # Format date
    force=True)
logger = logging.getLogger()

def get_main_df():
    """
    Get data frame from wandb
    Covert to same format we used for neural network
    """
    run = wandb.init(project=args.project_name)
    logger.info("Downloading data artifact")
    artifact = run.use_artifact(args.main_df, type='preprocessed_data')
    artifact_path = artifact.file()
    df = pd.read_parquet(artifact_path)

    # Encoding categorical data
    user_ids = df["user_id"].unique().tolist()
    anime_ids = df["anime_id"].unique().tolist()

    # Dicts of format {id: count_number}
    anime_to_index = {value: count for count, value in enumerate(anime_ids)}
    user_to_index = {value: count for count, value in enumerate(user_ids)}

    # Dicts of format {count_number: id}
    index_to_user = {count: value for count, value in enumerate(user_ids)}
    # index_to_anime = {count: value for count, value in enumerate(anime_ids)}

    # Convert values of format id to count_number
    df["user"] = df["user_id"].map(user_to_index)
    df["anime"] = df["anime_id"].map(anime_to_index)
    df = df[['user', 'anime', 'rating']]
    df = df.sample(frac=1, random_state=42)

    return df, user_to_index, index_to_user


def get_model():
    run = wandb.init(project=args.project_name)
    logger.info("Downloading model")
    artifact = run.use_artifact(args.model, type='h5')
    artifact_path = artifact.file()
    model = tf.keras.models.load_model(artifact_path)
    return model

def get_weights():
    logger.info("Getting weights")
    model = get_model()
    anime_weights = model.get_layer('anime_embedding')
    anime_weights = anime_weights.get_weights()[0]
    anime_weights = anime_weights / np.linalg.norm(
        anime_weights, axis=1).reshape((-1, 1))

    user_weights = model.get_layer('user_embedding')
    user_weights = user_weights.get_weights()[0]
    user_weights = user_weights / np.linalg.norm(
        user_weights, axis=1).reshape((-1, 1))
    logger.info("Weights extracted!")
    return anime_weights, user_weights

def get_random_user(df):
    ratings = df.groupby('user_id').size()
    random_user = ratings[ratings < int(args.max_ratings)].sample(1).index[0]
    return random_user


pd.set_option("max_colwidth", None)

def find_similar_users(item=None, n=args.id_query_number):
    rating_df, user_to_index, index_to_user = get_main_df()
    anime_weights, user_weights = get_weights()
    weights = user_weights

    if item is None:
        item = get_random_user(rating_df)
    else:
        item=item

    try:
        index = item
        encoded_index = user_to_index.get(index)
    
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)
        
        SimilarityArr = []
        
        for close in closest:
            similarity = dists[close]

            if isinstance(item, int):
                decoded_id = index_to_user.get(close)
                SimilarityArr.append({"similar_users": decoded_id, 
                                      "similarity": similarity})

        Frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity", 
                                                        ascending=False)
        
        return Frame
    
    except:
        print('{}!, Not Found in User list'.format(name))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an anime recommendation neural network",
        fromfile_prefix_chars="@",
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

    parser.add_argument(
        "--project_name",
        type=str,
        help="Name of wandb project",
        required=True
    )

    parser.add_argument(
        "--main_df",
        type=str,
        help="Main prerpocessed data frame",
        required=True
    )

    parser.add_argument(
        "--sypnopses_df",
        type=str,
        help="Sypnopses df",
        required=True
    )

    parser.add_argument(
        "--anime_df",
        type=str,
        help="anime df",
        required=True
    )

    parser.add_argument(
        "--user_query",
        type=str,
        help="list of input names of anime to query",
        required=True
    )

    parser.add_argument(
        "--id_query_number",
        type=str,
        help="Number of similar users to return",
        required=True
    )

    parser.add_argument(
        "--max_ratings",
        type=str,
        help="Maximum ratings you want to user to have made",
        required=True
    )

    parser.add_argument(
        "--random_user",
        type=lambda x: bool(strtobool(x)),
        help="Decide whether or not to use a random user id",
        required=True
    )

    args = parser.parse_args()
    go(args)