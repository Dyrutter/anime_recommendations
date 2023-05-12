import argparse
import logging
# import os
import wandb
# import string
import pandas as pd
import numpy as np
# import tensorflow as tf
from distutils.util import strtobool
import random
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt


logging.basicConfig(
    filename='./user_prefs.log',  # Path to log file
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

    df["user"] = df["user_id"].map(user_to_index)
    df["anime"] = df["anime_id"].map(anime_to_index)
    df = df[['user', 'anime', 'rating', 'user_id', 'anime_id']]
    df = df.sample(frac=1, random_state=42)
    return df, user_to_index, index_to_user


def get_genres(anime_df):
    """
    Get a list of all possible anime genres
    """
    genres = anime_df['Genres'].unique().tolist()
    # Get genres individually (instances have lists of genres)
    possibilities = list(set(str(genres).split()))
    # Remove non alphanumeric characters
    possibilities = sorted(
        list(set([re.sub(r'[\W_]', '', e) for e in possibilities])))
    # Convert incomplete categories to their proper names
    rem = ['Slice', "of", "Life", "Martial", "Arts", "Super", "Power", 'nan']
    fixed = possibilities + ['SliceofLife', 'SuperPower', 'MartialArts']
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


def get_random_user():
    """
    Get a random user from main data frame
    """
    rating_df, user_to_index, index_to_user = get_main_df()
    possible_users = list(user_to_index.keys())

    random_user = int(random.choice(possible_users))
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
    logger.info("Downloading anime data artifact")
    artifact = run.use_artifact(args.anime_df, type='raw_data')
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
                 "Source"]
    df = df[keep_cols]
    return df


def genre_cloud(anime_df, ID):
    genres = get_genres(anime_df)
    genres = (" ").join(list(map(str.upper, genres)))

    cloud = WordCloud(width=int(args.cloud_width),
                      height=int(args.cloud_height),
                      prefer_horizontal=0.85,
                      background_color='white',
                      contour_width=0.05,
                      colormap='spring')
    genres_cloud = cloud.generate(genres)
    fn = 'User_ID_' + str(ID) + '_genre_cloud.png'
    # genres_cloud.to_file(fp)
    return genres_cloud, fn


def source_cloud(anime_df, ID):
    source = get_sources(anime_df)
    sources = (" ").join(list(map(str.upper, source)))

    cloud = WordCloud(width=int(args.cloud_width),
                      height=int(args.cloud_height),
                      prefer_horizontal=0.85,
                      background_color='white',
                      contour_width=0.05,
                      colormap='spring')
    source_cloud = cloud.generate(sources)
    fn = 'User_ID_' + str(ID) + '_source_cloud.png'
    # source_cloud.to_file(fp)
    return source_cloud, fn


def show_cloud(cloud):
    plt.figure(figsize=(8, 6))
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def fave_genres(user, df, anime_df):
    watched = df[df.user_id == user]
    user_rating_percentile = np.percentile(
        watched.rating, float(args.favorite_percentile))
    watched = watched[watched.rating >= user_rating_percentile]
    top = (watched.sort_values(by="rating", ascending=False).anime_id.values)

    faves = anime_df[anime_df["anime_id"].isin(top)]
    faves = faves[["eng_version", "Genres"]]
    return pd.DataFrame(faves)  # added DataFrame to this


def fave_sources(user, df, anime_df):
    watched = df[df.user_id == user]
    user_rating_percentile = np.percentile(
        watched.rating, float(args.favorite_percentile))
    watched = watched[watched.rating >= user_rating_percentile]
    top = (watched.sort_values(by="rating", ascending=False).anime_id.values)

    faves = anime_df[anime_df["anime_id"].isin(top)]
    faves = faves[["eng_version", "Source"]]
    return pd.DataFrame(faves)


def fave_df(genres, sources):
    """
    Input source and genre dfs and returned merged df
    """
    genres = genres["Genres"]
    sources["Genres"] = genres
    return sources


def go(args):
    run = wandb.init(
        project=args.project_name,
        name="user_preferences")

    if args.random_user is True:
        user = get_random_user()
        logger.info("Using %s as random input user", user)
    else:
        user = int(args.user_query)

    df, user_to_index, index_to_user = get_main_df()
    anime_df = get_anime_df()

    genre_df = fave_genres(user, df, anime_df)
    source_df = fave_sources(user, df, anime_df)
    # genre_df = pd.DataFrame(favorite_genres)
    # source_df = pd.DataFrame(favorite_sources)

    genre_clouds, genre_fn = genre_cloud(genre_df, user)
    sources_cloud, source_fn = source_cloud(source_df, user)

    if args.show_clouds is True:  # FIND TIMER FOR PLT.SHOW
        show_cloud(genre_cloud)
        show_cloud(source_cloud)
    else:
        pass


# wordcloud.to_file("img/first_review.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get user preferences",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Wandb artifact with .h5 file of neural network",
        required=True
    )

    parser.add_argument(
        "--main_df",
        type=str,
        help="Main prerpocessed data frame",
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
        "--favorite_percentile",
        type=str,
        help="Top percentile to consider as favorite ratings",
        required=True
    )

    parser.add_argument(
        "--show_clouds",
        type=lambda x: bool(strtobool(x)),
        help="Boolean of whether or not to show word clouds at runtime",
        required=True
    )

    parser.add_argument(
        "--genre_fn",
        type=str,
        help="Genre image artifact name",
        required=True
    )

    parser.add_argument(
        "--source_fn",
        type=str,
        help="Sources image artifact name",
        required=True
    )

    parser.add_argument(
        "--cloud_width",
        type=str,
        help="Pixel width of word clouds",
        required=True
    )

    parser.add_argument(
        "--cloud_height",
        type=str,
        help="Pixel height of word clouds",
        required=True
    )

    parser.add_argument(
        "--prefs_csv",
        type=str,
        help="Artifact name of preferences csv file",
        required=True
    )

    args = parser.parse_args()
    go(args)
