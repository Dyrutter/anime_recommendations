import argparse
import logging
# import os
import wandb
import string
import pandas as pd
import numpy as np
import tensorflow as tf
from distutils.util import strtobool
import random


logging.basicConfig(
    filename='./similar_users.log',  # Path to log file
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
    artifact = run.use_artifact(args.main_df, type=args.main_df_type)
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

    # Convert values of format id to count_number
    df["user"] = df["user_id"].map(user_to_index)
    df["anime"] = df["anime_id"].map(anime_to_index)
    df = df[['user', 'anime', 'rating', 'user_id', 'anime_id']]
    df = df.sample(frac=1, random_state=42)

    return df, user_to_index, index_to_user


def get_model():
    run = wandb.init(project=args.project_name)
    logger.info("Downloading model")
    artifact = run.use_artifact(args.model, type=args.model_type)
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


def get_random_user():
    df, user_to_index, index_to_user = get_main_df()
    possible_users = list(user_to_index.keys())
    random_user = random.choice(possible_users)
    return random_user


def get_anime_name(anime_id, df):
    try:
        # Get a single anime from the anime df based on ID
        name = df[df.anime_id == anime_id].eng_version.values[0]
    except BaseException:
        raise ValueError("ID/eng_version pair was not found in data frame!")

    try:
        if name is np.nan:
            name = df[df.anime_id == anime_id].Name.values[0]
    except BaseException:
        raise ValueError("Name was not found in data frame!")
    return name


def get_anime_df():
    """
    Get data frame containing stats on each anime
    """
    run = wandb.init(project=args.project_name)
    artifact = run.use_artifact(args.anime_df, type=args.anime_df_type)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path)
    df = df.replace("Unknown", np.nan)

    df['anime_id'] = df['MAL_ID']
    df['japanese_name'] = df['Japanese name']
    df["eng_version"] = df['English name']
    df['eng_version'] = df.anime_id.apply(lambda x: get_anime_name(x, df))
    df.sort_values(by=['Score'],
                   inplace=True,
                   ascending=False,
                   kind='quicksort',
                   na_position='last')
    keep_cols = ["anime_id", "eng_version", "Score", "Genres", "Episodes",
                 "Premiered", "Studios", "japanese_name", "Name", "Type",
                 "Source", 'Rating', 'Members']
    df = df[keep_cols]
    return df


def get_fave_anime(df, anime_df, user, num_faves, TV_only):
    """
    Get a user's favorite anime. If more than one anime has the
    maxium rating, select the anime with the most episodes watched.
    Inputs:
        df: main pandas data frame
        anime_df: pandas data frame with anime statistics
        user: interger ID of the user in question
        num_faves: interger of favorite animes to return
        TV_only: Boolean of whether to only include TV shows as favorite anime
    Outputs:
        String of num_faves favorite anime
    """
    # Get the IDs of all anime a user has watched and narrow to max rated
    fave_df = df[df.user_id == user]
    fave_df = fave_df[fave_df.rating == max(fave_df.rating)]
    faves = fave_df['anime_id'].tolist()

    # Get names of each corresponding anime ID in the sorted list
    names = [anime_df[
        anime_df.anime_id == anime_id].Name.values[0] for anime_id in faves]

    # Get number of episodes of each corresponding anime ID
    eps = [anime_df[anime_df.anime_id == anime_id].Episodes.values[0]
           for anime_id in faves]

    fave_df['name'], fave_df['episode'] = names, eps
    # Convert episodes from type str to float
    fave_df['episodes'] = np.array(fave_df['episode']).astype(np.float32)

    try:
        # Get % of episodes the user has watched
        x = []
        for index, row in fave_df.iterrows():
            x.append(row['watched_episodes'] / row['episodes'])

        # Create new data frame of only rows with highest % watched
        fave_df["percent"] = x
        fave_df = fave_df[fave_df.percent == max(fave_df.percent)]

    except KeyError:
        logger.info("The data frame did not contain 'watched_episodes'")

    # Sort remaining data frame according to # of episodes if TV_only
    if TV_only is True:
        fave_df.sort_values(by='episodes', ascending=False, inplace=True)
    all_faves = fave_df['name'].tolist()
    try:
        # Return faves if enough exist, convert list to a string w/o brackets
        return str(all_faves[:num_faves])[1:-1]
    except IndexError:
        # If there aren't enough qualifying anime, return all faves
        return str(all_faves)[1:-1]


pd.set_option("max_colwidth", None)


def find_similar_users(user_id, n_users, num_faves, TV_only):
    df, user_to_index, index_to_user = get_main_df()
    anime_weights, user_weights = get_weights()
    anime_df = get_anime_df()
    weights = user_weights

    # Specify filename here in case the a random ID is used
    filename = 'User_' + str(user_id).translate(
        {ord(c): None for c in string.whitespace}) + '.csv'

    index = user_id
    encoded_index = user_to_index.get(index)

    dists = np.dot(weights, weights[encoded_index])
    sorted_dists = np.argsort(dists)
    n_users = n_users + 1
    closest = sorted_dists[-n_users:]

    SimilarityArr = []

    for close in closest:
        similarity = dists[close]
        decoded_id = index_to_user.get(close)
        if decoded_id != user_id:
            faves = get_fave_anime(
                df, anime_df, decoded_id, num_faves, TV_only)
            SimilarityArr.append(
                {"similar_users": decoded_id,
                 "similarity": similarity,
                 "favorite_animes": faves})

    Frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity",
                                                    ascending=False)

    return Frame, filename, user_id


def go(args):
    # Initialize run
    run = wandb.init(
        project=args.project_name,
        name="similar_users")

    if args.random_user is True:
        user_id = get_random_user()
        logger.info("Using random user ID %s", user_id)
    else:
        user_id = int(args.sim_user_query)

    # Create data frame file
    df, filename, user_id = find_similar_users(int(user_id), int(
        args.id_query_number), int(args.num_faves), args.TV_only)
    df.to_csv(filename, index=False)

    # Create similar users artifact
    logger.info("Creating similar users artifact")
    description = "Users most similar to: " + str(args.sim_user_query)
    artifact = wandb.Artifact(
        name=args.sim_users_fn,
        type=args.sim_users_type,
        description=description,
        metadata={"Queried user: ": user_id, "Filename: ": filename})
    # Upload to wandb
    artifact.add_file(filename)
    logger.info("Logging similar users artifact")
    run.log_artifact(artifact)
    artifact.wait()

    # Save the user ID as an artifact
    ID_df = pd.DataFrame([user_id], columns=['User_ID'])
    logger.info("Creating user_id artifact")
    description = "User ID queried, will be-re-used in further steps"
    fn = str(user_id) + '.csv'
    ID_df.to_csv(fn, index=False)
    artifact = wandb.Artifact(
        name=args.ID_fn,
        type=args.ID_type,
        description=description,
        metadata={"Queried user ": user_id, "Filename ": fn})
    artifact.add_file(fn)
    logger.info("Logger user id")
    run.log_artifact(artifact)
    artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get similar users",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--anime_df",
        type=str,
        help="Artifact name of anime data df",
        required=True
    )

    parser.add_argument(
        "--anime_df_type",
        type=str,
        help="Artifact type of anime df",
        required=True
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Wandb artifact of neural network",
        required=True
    )

    parser.add_argument(
        "--model_type",
        type=str,
        help="Artifact type of model",
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
        "--main_df_type",
        type=str,
        help="Type of main data frame",
        required=True
    )

    parser.add_argument(
        "--sim_user_query",
        type=str,
        help="input user id to query",
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

    parser.add_argument(
        "--num_faves",
        type=str,
        help="Number of a similar user's favorite anime to return",
        required=True
    )

    parser.add_argument(
        "--TV_only",
        type=lambda x: bool(strtobool(x)),
        help="Boolean of whether to return a similar user's TV for type",
        required=True
    )

    parser.add_argument(
        "--sim_users_fn",
        type=str,
        help="Filename of similar users artifact",
        required=True
    )

    parser.add_argument(
        "--sim_users_type",
        type=str,
        help="Type of similar users artifact",
        required=True
    )

    parser.add_argument(
        "--ID_fn",
        type=str,
        help="filename of artifact containing the user ID that was queried",
        required=True
    )

    parser.add_argument(
        "--ID_type",
        type=str,
        help='artifact type user ID was saved as',
        required=True
    )

    args = parser.parse_args()
    go(args)
