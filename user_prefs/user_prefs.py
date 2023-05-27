import argparse
import logging
import os
import wandb
import random
import re
import pandas as pd
import numpy as np
from distutils.util import strtobool
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


def main_df_by_id():
    """
    Get main df from wandb and covert to same format used for neural network
    Outputs:
        df: Main data frame with added columns containing additional indices
        user_to_index: enumerated IDs of format {ID: enumerated_index}
        index_to_user: enumerated IDs of format {enumerated_index: ID}
    """
    run = wandb.init(project=args.project_name)
    artifact = run.use_artifact(args.main_df, type=args.main_df_type)
    artifact_path = artifact.file()
    df = pd.read_parquet(artifact_path)
    n_ratings = df['user_id'].value_counts(dropna=True)
    df = df[df['user_id'].isin(
        n_ratings[n_ratings >= int(400)].index)].copy()

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


def get_anime_name(anime_id, df):
    """
    Helper function for loading anime data frame
    """
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


def genre_cloud(anime_df, ID):
    """
    Create a word cloud of a user's favorite genres
    Inputs:
        anime_df: anime stats data frame
        ID: User ID to create cloud of
    Outputs:
        genres_cloud: A wordcloud object of the user's favorite genres
        fn: Filename wordcloud was saved as
    If args.show_cloud is True, cloud will show at runtime
    """
    genres = get_genres(anime_df)
    genres = (" ").join(list(map(str.upper, genres)))

    cloud = WordCloud(width=int(args.cloud_width),
                      height=int(args.cloud_height),
                      prefer_horizontal=0.85,
                      background_color='white',
                      contour_width=0.05,
                      colormap='spring')
    genres_cloud = cloud.generate(genres)
    fn = "User_ID_" + str(ID) + '_' + args.genre_fn
    genres_cloud.to_file(fn)
    if args.show_clouds is True:
        show_cloud(genres_cloud)
    return genres_cloud, fn


def source_cloud(anime_df, ID):
    """
    Create a word cloud of a user's favorite sources
    Inputs:
        anime_df: anime stats data frame
        ID: User ID to create cloud of
    Outputs:
        source_cloud: a wordcloud object of the user's favorite sources
        fn: The filename of the word cloud
    If args.show_cloud is True, cloud will show at runtime
    """
    source = get_sources(anime_df)
    sources = (" ").join(list(map(str.upper, source)))

    cloud = WordCloud(width=int(args.cloud_width),
                      height=int(args.cloud_height),
                      prefer_horizontal=0.85,
                      background_color='white',
                      contour_width=0.05,
                      colormap='spring')
    source_cloud = cloud.generate(sources)
    fn = 'User_ID_' + str(ID) + '_' + args.source_fn
    source_cloud.to_file(fn)
    if args.show_clouds is True:
        show_cloud(source_cloud)
    return source_cloud, fn


def get_genres(anime_df):
    """
    Get all possible anime genres
    Input: data frame containing anime statistics
    Output: All possible anime genres in list format
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
    Get all possible anime sources.
    Input: data frame containing anime statistics
    Output: All possible anime sources in list format
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


def show_cloud(cloud):
    """
    Helper function to desplay a word cloud.
    Input: Either a genre or sources word cloud
    """
    fig = plt.figure(figsize=(8, 6))
    timer = fig.canvas.new_timer(interval=int(args.interval))
    timer.add_callback(plt.close)
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    timer.start()
    plt.show()


def get_random_user(df, user_to_index, index_to_user):
    """
    Get a random user from all possible users
    Inputs:
        df: main df of cols ['user', 'anime', 'rating', 'user_id', 'anime_id']
        user_to_index: enumerated IDs of format {ID: enumerated_index}
        index_to_user: enumerated IDs of format {enumerated_index: ID}
    Outputs:
        random_user: Interger value of a random user in df
    """
    possible_users = list(user_to_index.keys())
    random_user = random.choice(possible_users)
    return random_user


def fave_genres(user, df, anime_df):
    """
    Get the favorite genres of a user
    Inputs:
        user: an integer user ID
        df: main df of cols ['user', 'anime', 'rating', 'user_id', 'anime_id']
        anime_df: df containing anime statisitics
    Outputs:
        faves: Pandas data frame containing a user's favorite genres
    """
    watched = df[df.user_id == user]
    user_rating_percentile = np.percentile(
        watched.rating, float(args.favorite_percentile))
    watched = watched[watched.rating >= user_rating_percentile]
    top = (watched.sort_values(by="rating", ascending=False).anime_id.values)

    faves = anime_df[anime_df["anime_id"].isin(top)]
    faves = faves[["eng_version", "Genres"]]
    return pd.DataFrame(faves)


def fave_sources(user, df, anime_df):
    """
    Get the favorite sources of a user
    Inputs:
        user: an integer user ID
        df: main df of cols ['user', 'anime', 'rating', 'user_id', 'anime_id']
        anime_df: df containing anime statisitics
    Outputs:
        faves: Pandas data frame containing a user's favorite sources
    """
    watched = df[df.user_id == user]
    user_rating_percentile = np.percentile(
        watched.rating, float(args.favorite_percentile))
    watched = watched[watched.rating >= user_rating_percentile]
    top = (watched.sort_values(by="rating", ascending=False).anime_id.values)

    faves = anime_df[anime_df["anime_id"].isin(top)]
    faves = faves[["eng_version", "Source"]]
    return pd.DataFrame(faves)


def get_fave_df(genres, sources, ID, save):
    """
    Merge data frames of a user's favorite genres and sources
    Inputs:
        genres: Pandas data frame containing a user's favorite genres
        sources: Pandas data frame containing a user's favorite sources
        ID: The user ID to be queried
        save: Boolean of whether or not to save the data frame locally
    Outputs:
        sources: Merged Pandas data frame of favorite sources and genres
        fn: Filename data frame is saved as
    """
    genres = genres["Genres"]
    sources["Genres"] = genres
    if save is True:
        fn = 'User_ID_' + str(ID) + '_' + args.prefs_csv
        sources.to_csv(fn)
        return sources, fn
    else:
        return sources


def get_ID_artifact():
    """
    Get the user ID artifact and return its integer value
    """
    run = wandb.init(project=args.project_name)
    artifact = run.use_artifact(args.flow_user, type=args.ID_type)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path)
    return df.values[0][0]


def go(args):
    # Initiate run and get data frames
    run = wandb.init(project=args.project_name)
    df, user_to_index, index_to_user = main_df_by_id()
    anime_df = get_anime_df()

    # Establish which user to find preferences of
    if args.random_user is True:
        user = get_random_user(df, user_to_index, index_to_user)
        logger.info("Using %s as random input user", user)
    elif args.use_local_user is True:
        user = int(args.user_query)
        logger.info("Using locally specified user %s as user", user)
    elif args.flow_user is True:
        user = get_ID_artifact
        logger.info("Using ID artifact %s as local user", user)
    else:
        logger.info("NO ID WAS INPUT")
        raise ValueError("NO ID WAS INPUT")

    # Get the user's favorite genres and sources
    genre_df = fave_genres(user, df, anime_df)
    source_df = fave_sources(user, df, anime_df)

    # Create genre cloud, source cloud, and preferences csv file
    genres_cloud, genre_fn = genre_cloud(genre_df, user)
    sources_cloud, source_fn = source_cloud(source_df, user)
    fave_df, fave_fn = get_fave_df(
        genre_df, source_df, user, args.save_faves)

    # Log favorite genre cloud
    logger.info("Genre Cloud artifact")
    genre_cloud_artifact = wandb.Artifact(
        name=args.genre_fn,
        type=args.cloud_type,
        description='Cloud image of favorite genres',
        metadata={"ID": user, "Filename": genre_fn})
    genre_cloud_artifact.add_file(genre_fn)
    run.log_artifact(genre_cloud_artifact)
    logger.info("Genre cloud logged!")
    genre_cloud_artifact.wait()

    # Log favorite source cloud
    logger.info("creating source cloud artifact")
    source_cloud_artifact = wandb.Artifact(
        name=args.source_fn,
        type=args.cloud_type,
        description='Image of source cloud',
        metadata={"ID": user, "Filename": source_fn})
    source_cloud_artifact.add_file(source_cloud_artifact)
    run.log_artifact(source_cloud_artifact)
    logger.info('Source cloud logged!')
    source_cloud_artifact.wait()

    # Log favorites csv file
    logger.info("Creating favorites csv")
    favorites_artifact = wandb.Artifact(
        name=args.prefs_csv,
        type=args.fave_art_type,
        description='Csv file of a users favorite Genres and sources',
        metadata={"ID": user, "Filename": fave_fn})
    favorites_artifact.add_file(fave_fn)
    run.log_artifact(favorites_artifact)
    logger.info("Favorites data frame logged!")
    favorites_artifact.wait()

    if args.save_faves is False:
        os.remove(genre_fn)
        os.remove(source_fn)
        os.remove(fave_fn)


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

    parser.add_argument(
        "--interval",
        type=str,
        help="Interval in milliseconds to display clouds",
        required=True
    )

    parser.add_argument(
        "--save_faves",
        type=lambda x: bool(strtobool(x)),
        help="Whether or not to save clouds and fave csv file locally",
        required=True
    )

    parser.add_argument(
        "--flow_user",
        type=str,
        help="Latest user ID artifact created for use in MLflow runs",
        required=True
    )

    parser.add_argument(
        "--from_flow",
        type=lambda x: bool(
            strtobool(x)),
        help="Bool of whether to use the ID Artifact created using MLflow",
        required=True)

    parser.add_argument(
        "--use_local_user",
        type=lambda x: bool(
            strtobool(x)),
        help="Whether to use user_query in config file instead of artifact",
        required=True)

    parser.add_argument(
        "--main_df_type",
        type=str,
        help="Type of main data frame",
        required=True
    )

    parser.add_argument(
        "--anime_df_type",
        type=str,
        help="Artifact type of anime df",
        required=True
    )

    parser.add_argument(
        "--ID_type",
        type=str,
        help='artifact type user ID was saved as',
        required=True
    )

    parser.add_argument(
        "--cloud_type",
        type=str,
        help="Artifact type of favorites clouds",
        required=True
    )

    parser.add_argument(
        "--fave_art_type",
        type=str,
        help="Artifact type of favorites csv",
        required=True
    )

    args = parser.parse_args()
    go(args)
