import argparse
import logging
import os
import wandb
import random
import pandas as pd
import numpy as np
from distutils.util import strtobool
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import defaultdict


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
    # Load data frame artifact
    run = wandb.init(project=args.project_name)
    artifact = run.use_artifact(args.main_df, type=args.main_df_type)
    artifact_path = artifact.file()
    df = pd.read_parquet(artifact_path)

    # Narrow df to users with > 400 ratings
    n_ratings = df['user_id'].value_counts(dropna=True)
    df = df[df['user_id'].isin(n_ratings[n_ratings >= int(400)].index)].copy()

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
    Load data frame artifact containing info on each anime from wandb.
    Output:
        df: Pandas Data Frame containing all anime and their relevant stats
    """
    run = wandb.init(project=args.project_name)
    artifact = run.use_artifact(args.anime_df, type=args.anime_df_type)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path)
    df = df.replace("Unknown", np.nan)

    df['anime_id'], df['eng_version'] = df['MAL_ID'], df['English name']
    df['eng_version'] = df.anime_id.apply(lambda x: get_anime_name(x, df))
    keep_cols = ["anime_id", "eng_version", "Genres", "Name", "Type",
                 "Source", "Rating"]
    df = df[keep_cols]
    return df


def get_anime_name(anime_id, df):
    """
    Helper function for loading anime data frame
    Inputs:
        anime_id: The ID of an anime
        df: data frame containing all anime, taken from get_anime_df()
    Outputs:
        name: The english name of anime_id
    """
    name = df[df.anime_id == anime_id].Name.values[0]
    return name


def get_genres(anime_df):
    """
    Get individual anime genres in ["Genres"], repeated per anime. Currently
    every item in anime_df's ["Genres"] is a list of genres. For example, the
    anime "Gosick" has ["Genres"]: [Mystery, Historical, Drama, Romance]. This
    Function takes "Mystery", "Historical", "Drama", and "Romance" and appends
    them to a larger list which will be used to assess frequency
    Input:
        anime_df: data frame containing all anime, taken from get_anime_df()
    Output:
        genres_list: Individual anime genres in anime_df's ["Genres"],
            repeated per anime and merged into a single list
        all_genres: A default dict of all genres in the data frame
    """
    anime_df.dropna(inplace=False)
    all_genres = defaultdict(int)

    genres_list = []
    for genres in anime_df['Genres']:
        if isinstance(genres, str):
            for genre in genres.split(','):
                genres_list.append(genre)
                all_genres[genre.strip()] += 1
    return genres_list, all_genres


def get_sources(anime_df):
    """
    Get individual anime sources in ["Sources"], repeated per anime. Currently
    every item in anime_df's ["Sources"] is a list of sources. This function
    appends each source to a list of all sources for every anime.
    Input:
        anime_df: data frame containing all anime, taken from get_anime_df()
    Output:
        sources_list: Individual anime sources in anime_df's ["Sources"],
            repeated per anime and merged into a single list
        all_sources: A defaultdict object of all sources
    """
    anime_df.dropna(inplace=False)
    all_sources = defaultdict(int)
    sources_list = []
    for sources in anime_df['Source']:
        if isinstance(sources, str):
            for source in sources.split(','):
                sources_list.append(source)
                all_sources[source.strip()] += 1
    return sources_list, all_sources


def genre_cloud(anime_df, ID):
    """
    Create a word cloud of a user's favorite genres
    Inputs:
        anime_df: data frame containing all anime, taken from get_anime_df()
        ID: User ID to create cloud of
    Outputs:
        genres_cloud: A wordcloud object of the user's favorite genres
        fn: Filename wordcloud was saved as
    If args.show_cloud is True, cloud will show at runtime
    """
    genres, genre_dict = get_genres(anime_df)
    cloud = WordCloud(width=int(args.cloud_width),
                      height=int(args.cloud_height),
                      prefer_horizontal=0.85,
                      background_color='white',
                      contour_width=0.05,
                      colormap='spring')
    genres_cloud = cloud.generate_from_frequencies(genre_dict)
    fn = "User_ID_" + str(ID) + '_' + args.genre_fn
    genres_cloud.to_file(fn)
    return genres_cloud, fn


def source_cloud(anime_df, ID):
    """
    Create a word cloud of a user's favorite sources
    Inputs:
        anime_df: data frame containing all anime, taken from get_anime_df()
        ID: User ID to create source preferences cloud of
    Outputs:
        source_cloud: a wordcloud object of the user's favorite sources
        fn: The filename of the word cloud
    If args.show_cloud is True, cloud will show at runtime
    """
    sources, source_dict = get_sources(anime_df)
    cloud = WordCloud(width=int(args.cloud_width),
                      height=int(args.cloud_height),
                      prefer_horizontal=0.85,
                      background_color='gray',
                      contour_width=0.05,
                      colormap='autumn')
    source_cloud = cloud.generate_from_frequencies(source_dict)
    fn = 'User_ID_' + str(ID) + '_' + args.source_fn
    source_cloud.to_file(fn)
    return source_cloud, fn


def show_cloud(cloud):
    """
    Helper function to desplay a word cloud.
    Input:
        cloud: Either a genre or sources word cloud object
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


def get_fave_df(genres, sources, ID):
    """
    Merge data frames of a user's favorite genres and sources
    Inputs:
        genres: Pandas data frame containing a user's favorite genres
        sources: Pandas data frame containing a user's favorite sources
        ID: The user ID to be queried
    Outputs:
        sources: Merged Pandas data frame of favorite sources and genres
        fn: Filename data frame is saved as
    """
    genres = genres["Genres"]
    sources["Genres"] = genres
    fn = 'User_ID_' + str(ID) + '_' + args.prefs_csv
    sources.to_csv(fn)
    return sources, fn


def get_ID_artifact():
    """
    Get the user ID data frame artifact
    Output:
        ID: Int, ID that was saved as an artifact for MLflow use
    """
    run = wandb.init(project=args.project_name)
    artifact = run.use_artifact(args.flow_user, type=args.ID_type)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path)
    return df.values[0][0]


def select_user(df, user_to_index, index_to_user):
    """
    Choose user to analyze based. Will be either the ID used
        in the MLflow workflow if args.prefs_from_flow is True, the ID input
        in the config file under the section ["Users"]["prefs_user_query"] if
        args.pefs_local_user is True, or a random ID
    Inputs:
        df: Main df taken from main_df_by_id()
        user_to_index: dict, enumerated mapping taken from main_df_by_id()
        index_to_user: dict, enumerated mapping taken from main_df_by_id()
    Outputs:
        user: Integer of the user ID to analzye
        Type: The type of user (Artifact, local, or random) for metadata use
    """
    if args.prefs_from_flow is True:
        user = get_ID_artifact()
        Type = "MLflow ID"
        logger.info("Using %s as input use taken from MLflow", user)
        return user, Type

    elif args.prefs_local_user is True:
        user = int(args.prefs_user_query)
        Type = "Local Config File ID"
        logger.info("Using %s as config file-specified input user", user)
        return user, Type
    else:
        user = get_random_user(df, user_to_index, index_to_user)
        Type = "Random User"
        logger.info("Using %s as random input user", user)
        return user, Type


def go(args):
    # Initiate run and get data frames
    run = wandb.init(project=args.project_name)
    df, user_to_index, index_to_user = main_df_by_id()
    anime_df = get_anime_df()
    user, Type = select_user(df, user_to_index, index_to_user)

    # Get the user's favorite genres and sources
    genre_df = fave_genres(user, df, anime_df)
    source_df = fave_sources(user, df, anime_df)

    # Create genre cloud, source cloud, and preferences csv file
    genres_cloud, genre_fn = genre_cloud(genre_df, user)
    sources_cloud, source_fn = source_cloud(source_df, user)
    fave_df, fave_fn = get_fave_df(genre_df, source_df, user)

    # Log favorite genre cloud
    logger.info("Genre Cloud artifact")
    genre_cloud_artifact = wandb.Artifact(
        name=args.genre_fn,
        type=args.cloud_type,
        description='Cloud image of favorite genres',
        metadata={"ID": user, "User_type": Type, "Filename": genre_fn})
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
        metadata={"ID": user, "User_Type": Type, "Filename": source_fn})
    source_cloud_artifact.add_file(source_fn)
    run.log_artifact(source_cloud_artifact)
    logger.info('Source cloud logged!')
    source_cloud_artifact.wait()

    # Log favorites csv file
    logger.info("Creating favorites csv")
    favorites_artifact = wandb.Artifact(
        name=args.prefs_csv,
        type=args.fave_art_type,
        description='Csv file of a users favorite Genres and sources',
        metadata={"ID": user, "User_Type": Type, "Filename": fave_fn})
    favorites_artifact.add_file(fave_fn)
    run.log_artifact(favorites_artifact)
    logger.info("Favorites data frame logged!")
    favorites_artifact.wait()

    if args.show_clouds is True:
        show_cloud(genres_cloud)
        show_cloud(sources_cloud)

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
        "--prefs_user_query",
        type=str,
        help="input user id to query",
        required=True
    )

    parser.add_argument(
        "--pref_random_user",
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
        "--prefs_from_flow",
        type=lambda x: bool(
            strtobool(x)),
        help="Bool of whether to use the ID Artifact created using MLflow",
        required=True)

    parser.add_argument(
        "--prefs_local_user",
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
