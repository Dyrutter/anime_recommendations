import logging
import wandb
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import re


logging.basicConfig(
    filename='./load.log',  # Path to log file
    level=logging.INFO,  # Log info, warnings, errors, and critical errors
    filemode='a',  # Create log file if one doesn't already exist and add
    format='%(asctime)s-%(name)s - %(levelname)s - %(message)s',
    datefmt='%d %b %Y %H:%M:%S %Z',  # Format date
    force=True)
logger = logging.getLogger()


def get_model(project='anime_recommendations',
              model='wandb_anime_nn.h5:v6',
              artifact_type='h5'):
    run = wandb.init(project=project)  # project=args.project_name)
    logger.info("Downloading model")
    # args.model, type='h5')
    artifact = run.use_artifact(model, type=artifact_type)
    artifact_path = artifact.file()
    model = tf.keras.models.load_model(artifact_path)
    return model


def get_weights(model):
    logger.info("Getting weights")
    # model = get_model() ###### REMEMBER TO UPDATE THIS LINE IN ALL FUNCTIONS
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


def get_sypnopses_df(
        project='anime_recommendations',
        sypnopsis_df="sypnopses_artifact_latest",
        artifact_type='raw_data'):
    """
    Download sypnopses df from wandb
    """
    run = wandb.init(project=project)  # args.project_name)
    logger.info("Downloading sypnopses df")
    # args.sypnopses_df, type='raw_data')
    artifact = run.use_artifact(sypnopsis_df, type=artifact_type)
    artifact_path = artifact.file()
    cols = ["MAL_ID", "Name", "Genres", "sypnopsis"]
    df = pd.read_csv(artifact_path, usecols=cols)
    logger.info("Sypnopsis df shape is %s", df.shape)
    return df


def get_anime_df(
        project='anime_recommendations',
        anime_df='all_anime.csv:latest',
        artifact_type='raw_data'):
    """
    Get data frame containing stats on each anime
    """
    run = wandb.init(project=project)  # args.project_name)
    logger.info("Downloading anime data artifact")
    # args.anime_df, type='raw_data')
    artifact = run.use_artifact(anime_df, type=artifact_type)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path)
    logger.info("Orignal anime df shape is %s", df.shape)
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
    logger.info("Final anime df shape is %s", df.shape)
    return df


def get_random_user(rating_df, user_to_index, index_to_user):
    """
    Get a random user from main data frame
    """
    possible_users = list(user_to_index.keys())

    random_user = int(random.choice(possible_users))
    return random_user


def main_df_by_anime(
        project='anime_recommendations',
        main_df='preprocessed_stats.parquet:v2',
        artifact_type='preprocessed_data'):
    """
    Get data frame from wandb
    Covert to same format we used for neural network
    """

    run = wandb.init(project=project)  # project=args.project_name)
    logger.info("Downloading data artifact")
    # args.main_df, type='preprocessed_data')
    artifact = run.use_artifact(main_df, type=artifact_type)
    artifact_path = artifact.file()
    df = pd.read_parquet(artifact_path)
    logger.info("Main preprocessed df shape is %s", df.shape)

    # Encoding categorical data
    user_ids = df["user_id"].unique().tolist()
    anime_ids = df["anime_id"].unique().tolist()

    # Dicts of format {id: count_number}
    anime_to_index = {value: count for count, value in enumerate(anime_ids)}
    user_to_index = {value: count for count, value in enumerate(user_ids)}

    # Dicts of format {count_number: id}
    index_to_anime = {count: value for count, value in enumerate(anime_ids)}

    # Convert values of format id to count_number
    df["user"] = df["user_id"].map(user_to_index)
    df["anime"] = df["anime_id"].map(anime_to_index)
    df = df[['user', 'anime', 'rating']]
    df = df.sample(frac=1, random_state=42)

    logger.info("Final preprocessed df shape is %s", df.shape)
    logger.info("Final preprocessed df head is %s", df.head())

    return df, anime_to_index, index_to_anime


def main_df_by_id(
        project='anime_recommendations',
        main_df='preprocessed_stats.parquet:v2',
        artifact_type='preprocessed_data'):
    """
    Get data frame from wandb
    Covert to same format we used for neural network
    """
    run = wandb.init(project=project)  # project=args.project_name)
    logger.info("Downloading data artifact")
    # args.main_df, type='preprocessed_data')
    artifact = run.use_artifact(main_df, type=artifact_type)
    artifact_path = artifact.file()
    df = pd.read_parquet(artifact_path)
    n_ratings = df['user_id'].value_counts(dropna=True)
    df = df[df['user_id'].isin(
        n_ratings[n_ratings >= int(400)].index)].copy()

    # Encoding categorical data
    user_ids = df["user_id"].unique().tolist()
    # print (f'user ids lenght is {len(user_ids)}')
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


def get_anime_frame(anime, df):
    """
    Get either the anime's name or id as a data frame
    """
    if isinstance(anime, int):
        return df[df.anime_id == anime]
    if isinstance(anime, str):
        return df[df.eng_version == anime]


def get_sypnopsis(anime, sypnopsis_df):
    """
    Get sypnopsis of an anime from the sypnopsis data frame
    """
    if isinstance(anime, int):
        return sypnopsis_df[sypnopsis_df.MAL_ID == anime].sypnopsis.values[0]
    if isinstance(anime, str):
        return sypnopsis_df[sypnopsis_df.Name == anime].sypnopsis.values[0]


def get_genres(anime_df):
    """
    Get a list of all possible anime genres
    Input is data frame containing all anime
    """
    # anime_df = get_anime_df()
    genres = anime_df['Genres'].unique().tolist()
    # Get genres individually (instances have lists of genres)
    possibilities = list(set(str(genres).split()))
    # Remove non alphanumeric characters
    possibilities = sorted(
        list(set([re.sub(r'[\W_]', '', e) for e in possibilities])))
    # Convert incomplete categories to their proper names
    rem = ['Slice', "of", "Life", "Martial", "Arts", "Super", "Power", 'nan']
    fixed = possibilities + \
        ['Slice of Life', 'Super Power', 'Martial Arts', 'None']
    genre_list = sorted([i for i in fixed if i not in rem])
    return genre_list


def get_sources(anime_df):
    """
    Get a list of all possible anime genres
    """
    sources = anime_df['Source'].unique().tolist()
    # Get genres individually (instances have lists of genres)
    possibilities = list(set(str(sources).split()))
    # Remove non alphanumeric characters
    possibilities = sorted(list(
        set([re.sub(r'[\W_]', '', e) for e in possibilities])))

    remove = \
        ['novel', "Light", "Visual", "Picture", "Card", "game", "book", "Web"]
    fixed = possibilities + \
        ['LightNovel', 'VisualNovel', 'PictureBook', 'CardGame', "WebNovel"]
    source_list = sorted([i for i in fixed if i not in remove])
    return source_list
