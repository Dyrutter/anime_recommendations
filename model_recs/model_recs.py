import argparse
import logging
import os
import wandb
import string
import pandas as pd
from distutils.util import strtobool
import re
import random
import numpy as np
import tensorflow as tf

logging.basicConfig(
    filename='./model_recs.log',
    level=logging.INFO,
    filemode='a',
    format='%(asctime)s-%(name)s - %(levelname)s - %(message)s',
    datefmt='%d %b %Y %H:%M:%S %Z',
    force=True)
logger = logging.getLogger()


def get_full_df(
        project='anime_recommendations',
        main_df='preprocessed_stats.parquet:v2',
        artifact_type='preprocessed_data'):
    """
    Get data frame from wandb
    Covert to same format we used for neural network
    """
    run = wandb.init(project=project)
    # logger.info("Downloading data artifact")
    artifact = run.use_artifact(main_df, type=artifact_type)
    artifact_path = artifact.file()
    df = pd.read_parquet(artifact_path)

    # Encoding categorical data, get list of all anime ids and user ids
    user_ids = df["user_id"].unique().tolist()
    anime_ids = df["anime_id"].unique().tolist()

    # Dicts of format {id: count_number}
    anime_to_index = {value: count for count, value in enumerate(anime_ids)}
    user_to_index = {value: count for count, value in enumerate(user_ids)}

    # Convert values of format id to count_number
    df["user"] = df["user_id"].map(user_to_index)
    df["anime"] = df["anime_id"].map(anime_to_index)
    df = df[['user', 'anime', 'rating', 'anime_id', 'user_id']]
    df = df.sample(frac=1, random_state=42)
    return df


def get_anime_df(
        project='anime_recommendations',
        anime_df='all_anime.csv:latest',
        artifact_type='raw_data'):
    """
    Get data frame containing stats on each anime
    """
    run = wandb.init(project=project)
    # logger.info("Downloading anime data artifact")
    artifact = run.use_artifact(anime_df, type=artifact_type)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path)
    # logger.info("Orignal anime df shape is %s", df.shape)
    df = df.replace("Unknown", np.nan)

    df['anime_id'] = df['MAL_ID']
    df['japanese_name'] = df['Japanese name']
    df["eng_version"] = df['English name']
    # logger.info("Original English version is %s", df["eng_version"].head())
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
    # logger.info("Final anime df shape is %s", df.shape)
    return df


def get_anime_name(anime_id, df):
    name = df[df.anime_id == anime_id].eng_version.values[0]
    if name is np.nan:
        name = df[df.anime_id == anime_id].Name.values[0]
    return name


def get_unwatched(df, anime_df, user):
    watched = df[df.user_id == user]
    unwatched = anime_df[~anime_df['anime_id'].isin(watched.anime_ids.values)]
    anime_ids = df["anime_id"].unique().tolist()
    anime_to_index = {value: count for count, value in enumerate(anime_ids)}
    unwatched_ids = list(set(unwatched['anime_id']))
    return unwatched_ids.intersection(set(anime_to_index.keys()))


def get_user_anime_arr(df, anime_df, user, unwatched):
    user_ids = df["user_id"].unique().tolist()
    user_to_index = {value: count for count, value in enumerate(user_ids)}
    user_encoder = user_to_index.get(user)
    arr = np.hstack(([[user_encoder]] * len(unwatched), unwatched))
    return [arr[:, 0], arr[:, 1]]


def get_model(project='anime_recommendations',
              model='wandb_anime_nn.h5:v6',
              artifact_type='h5'):
    run = wandb.init(project=project)
    # logger.info("Downloading model")
    artifact = run.use_artifact(model, type=artifact_type)
    artifact_path = artifact.file()
    model = tf.keras.models.load_model(artifact_path)
    return model


def get_sypnopses_df(
        project='anime_recommendations',
        sypnopsis_df="sypnopses_artifact_latest",
        artifact_type='raw_data'):
    """
    Download sypnopses df from wandb
    """
    run = wandb.init(project=project)
    # logger.info("Downloading sypnopses df")
    artifact = run.use_artifact(sypnopsis_df, type=artifact_type)
    artifact_path = artifact.file()
    cols = ["MAL_ID", "Name", "Genres", "sypnopsis"]
    df = pd.read_csv(artifact_path, usecols=cols)
    # logger.info("Sypnopsis df shape is %s", df.shape)
    return df


def get_random_user(df):
    """
    Get a random user from main data frame
    """
    user_ids = df["user_id"].unique().tolist()
    user_to_index = {value: count for count, value in enumerate(user_ids)}
    possible_users = list(user_to_index.keys())
    random_user = int(random.choice(possible_users))
    return random_user


def get_sypnopsis(anime, sypnopsis_df):
    """
    Get sypnopsis of an anime from the sypnopsis data frame
    """
    if isinstance(anime, int):
        return sypnopsis_df[sypnopsis_df.MAL_ID == anime].sypnopsis.values[0]
    if isinstance(anime, str):
        return sypnopsis_df[sypnopsis_df.Name == anime].sypnopsis.values[0]


def recommendations(df, anime_df, syp_df, model, id_anime, unwatched, n_recs):
    anime_ids = df["anime_id"].unique().tolist()
    index_to_anime = {index: anime for index, anime in enumerate(anime_ids)}
    ratings = model.predict(id_anime).flatten()
    top_inds = (-ratings).argsort()[:n_recs]
    rec_ids = [index_to_anime.get(unwatched[x][0]) for x in top_inds]
    anime_recs, top_ids = [], []

    for index, anime_id in enumerate(unwatched):
        rating = ratings[index]
        ID = index_to_anime.get(anime_id[0])

        if ID in rec_ids:
            top_ids.append()
            try:
                condition = (anime_df.anime_id == ID)
                name = anime_df[condition]['eng_version'].values[0]
                genre = anime_df[condition]['Genres'].values[0]
                japanese_name = anime_df[condition]['japanese_name'].values[0]
                episodes = anime_df[condition]['Episodes'].values[0]
                premiered = anime_df[condition]['Premiered'].values[0]
                studios = anime_df[condition]["Studios"].values[0]
                score = anime_df[condition]["Score"].values[0]
                Type = anime_df[condition]["Type"].values[0]
                source = anime_df[condition]["Source"].values[0]
                anime_id = anime_df[condition]["anime_id"].values[0]
                sypnopsis = get_sypnopsis(int(anime_id), syp_df)
            except BaseException:
                continue
            anime_recs.append(
                {"Name": name, "Prediciton_rating": rating,
                 "Source": source, "Genres": genre, "anime_id": anime_id,
                 'Sypnopsis': sypnopsis, "Episodes": episodes,
                 "Japanese name": japanese_name, "Studios": studios,
                 "Premiered": premiered, "Score": score, "Type": Type})
    return pd.DataFrame(anime_recs).sort_values(
        by='Prediciton_rating', ascending=False)


def clean(item):
    """
    Pepare either a string or list of strings to csv format
    """
    translations = []
    if isinstance(item, list):
        for name in item:
            translated = str(name).translate(
                {ord(c): None for c in string.whitespace})
            translated = re.sub(r'\W+', '', translated)
            translations.append(translated.lower())
    else:
        translated = str(item).translate(
            {ord(c): None for c in string.whitespace})
        translations = re.sub(r'\W+', '', translated)
        translations = translations.lower()

    return translations


def go(args):
    # Initialize run
    run = wandb.init(project=args.project_name)
    df = get_full_df(
        project=args.project_name,
        main_df=args.main_df,
        artifact_type=args.main_df_type)
    anime_df = get_anime_df(
        project=args.project_name,
        anime_df=args.anime_df,
        artifact_type=args.all_artifact_type)
    sypnopsis_df = get_sypnopses_df(
        project=args.project_name,
        sypnopsis_df=args.sypnopsis_df,
        artifact_type=args.sypnopsis_type)
    model = get_model(
        project=args.project_name,
        model=args.model,
        artifact_type=args.model_type)

    if args.random_user is True:
        user = get_random_user(df)
        logger.info("Using %s as random input user", user)
    else:
        user = args.user_query
        logger.info("Using %s as input user", user)

    unwatched = get_unwatched(df, anime_df, user)
    arr = get_user_anime_arr(df, anime_df, user, unwatched)
    recs = recommendations(
        df,
        anime_df,
        sypnopsis_df,
        model,
        arr,
        unwatched,
        args.n_recs)
    filename = 'User_ID_' + str(user) + '_' + args.model_recs_fn
    recs.to_csv(filename, index=False)

    # Create artifact
    logger.info("Creating artifact")
    description = "Anime recs based on model rankings for user : " + str(user)
    artifact = wandb.Artifact(
        name=filename,
        type="csv",
        description=description,
        metadata={"Queried user: ": user})

    # Upload artifact to wandb
    artifact.add_file(filename)
    logger.info("Logging artifact for user %s", user)
    run.log_artifact(artifact)
    artifact.wait()

    if args.save_model_recs is False:
        os.remove(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get user preferences",
        fromfile_prefix_chars="@",
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
        help="Artifact type of main data frame",
        required=True
    )

    parser.add_argument(
        "--project_name",
        type=str,
        help="Name of wandb project",
        required=True
    )

    parser.add_argument(
        "--anime_df",
        type=str,
        help="anime df",
        required=True
    )

    parser.add_argument(
        "--anime_df_type",
        type=str,
        help='Artifact type of anime df',
        required=True
    )

    parser.add_argument(
        "--sypnopsis_df",
        type=str,
        help="Sypnopses df",
        required=True
    )

    parser.add_argument(
        "--sypnopsis_type",
        type=str,
        help="Sypnopsis artifact type",
        required=True
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model artifact",
        required=True
    )

    parser.add_argument(
        "--model_type",
        type=str,
        help="Type of neural network artifact, default h5",
        required=True
    )

    parser.add_argument(
        "--user_query",
        type=str,
        help="input user id to query",
        required=True
    )

    parser.add_argument(
        "--random_user",
        type=lambda x: bool(strtobool(x)),
        help="Decide whether or not to use a random user id",
        required=True
    )

    parser.add_argument(
        "--model_recs_fn",
        type=str,
        help="Artifact name of model recommendations csv file",
        required=True
    )

    parser.add_argument(
        "--save_model_recs",
        type=lambda x: bool(strtobool(x)),
        help="Whether or not to save model recs file locally",
        required=True
    )

    parser.add_argument(
        "--model_num_recs",
        type=str,
        help="Number of anime recommendations to return",
        required=True
    )

    args = parser.parse_args()
    go(args)