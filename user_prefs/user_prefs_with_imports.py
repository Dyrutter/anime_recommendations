from helper_functions.load import main_df_by_id, get_anime_df, get_sources
from helper_functions.load import get_random_user, get_genres
import logging
import os
import wandb
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))


logging.basicConfig(
    filename='./user_prefs.log',  # Path to log file
    level=logging.INFO,  # Log info, warnings, errors, and critical errors
    filemode='a',  # Create log file if one doesn't already exist and add
    format='%(asctime)s-%(name)s - %(levelname)s - %(message)s',
    datefmt='%d %b %Y %H:%M:%S %Z',  # Format date
    force=True)
logger = logging.getLogger()


def genre_cloud(anime_df, ID, show_clouds=True):
    genres = get_genres(anime_df)
    genres = (" ").join(list(map(str.upper, genres)))

    cloud = WordCloud(width=int(800),
                      height=int(600),
                      prefer_horizontal=0.85,
                      background_color='white',
                      contour_width=0.05,
                      colormap='spring')
    genres_cloud = cloud.generate(genres)
    fn = "User_ID_" + str(ID) + '_' + 'favorite_genres.png'
    genres_cloud.to_file(fn)
    if show_clouds is True:
        show_cloud(genre_cloud)
    return genres_cloud, fn


def source_cloud(anime_df, ID, show_clouds=True):
    source = get_sources(anime_df)
    sources = (" ").join(list(map(str.upper, source)))

    cloud = WordCloud(width=800,
                      height=600,
                      prefer_horizontal=0.85,
                      background_color='white',
                      contour_width=0.05,
                      colormap='spring')
    source_cloud = cloud.generate(sources)
    fn = 'User_ID_' + str(ID) + '_' + 'favorite_sources.png'
    source_cloud.to_file(fn)
    if show_clouds is True:
        show_cloud(source_cloud)
    return source_cloud, fn


def show_cloud(cloud):
    fig = plt.figure(figsize=(8, 6))
    timer = fig.canvas.new_timer(interval=int(3000))
    timer.add_callback(plt.close)
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    timer.start()
    plt.show()


def fave_genres(user, df, anime_df):
    watched = df[df.user_id == user]
    user_rating_percentile = np.percentile(
        watched.rating, float(80))
    watched = watched[watched.rating >= user_rating_percentile]
    top = (watched.sort_values(by="rating", ascending=False).anime_id.values)

    faves = anime_df[anime_df["anime_id"].isin(top)]
    faves = faves[["eng_version", "Genres"]]
    return pd.DataFrame(faves)


def fave_sources(user, df, anime_df):
    watched = df[df.user_id == user]
    user_rating_percentile = np.percentile(
        watched.rating, 80)
    watched = watched[watched.rating >= user_rating_percentile]
    top = (watched.sort_values(by="rating", ascending=False).anime_id.values)

    faves = anime_df[anime_df["anime_id"].isin(top)]
    faves = faves[["eng_version", "Source"]]
    return pd.DataFrame(faves)


def get_fave_df(genres, sources, ID, save=False):
    """
    Input source and genre dfs and returned merged df
    """
    genres = genres["Genres"]
    sources["Genres"] = genres
    if save is True:
        fn = 'User_ID_' + str(ID) + '_favorites.csv'
        sources.to_csv(fn)
        return sources, fn
    else:
        return sources


def go():
    run = wandb.init(
        project='anime_recommendations',
        name="user_preferences")

    df, user_to_index, index_to_user = main_df_by_id(
        project='anime_recommendations',
        main_df='preprocessed_stats.parquet:v2',
        artifact_type='preprocessed_data')

    anime_df = get_anime_df(
        project='anime_recommendations',
        anime_df="all_anime.csv:latest",
        artifact_type='raw_data')
    random_user = True

    if random_user is True:
        user = get_random_user(df, user_to_index, index_to_user)
        logger.info("Using %s as random input user", user)
    else:
        user = 109160

    genre_df = fave_genres(user, df, anime_df)
    source_df = fave_sources(user, df, anime_df)

    genres_cloud, genre_fn = genre_cloud(genre_df, user)
    sources_cloud, source_fn = source_cloud(source_df, user)
    fave_df, fave_fn = get_fave_df(
        genre_df, source_df, user, True)

    # Log favorite genre cloud
    logger.info("Genre Cloud artifact")
    genre_cloud_artifact = wandb.Artifact(
        name=genre_fn,
        type="image",
        description='Cloud image of favorite genres')
    genre_cloud_artifact.add_file(genre_fn)
    run.log_artifact(genre_cloud_artifact)
    logger.info("Genre cloud logged!")
    genre_cloud_artifact.wait()

    # Log favorite source cloud
    logger.info("creating source cloud artifact")
    source_cloud_artifact = wandb.Artifact(
        name=source_fn,
        type='cloud',
        description='Image of source cloud')
    source_cloud_artifact.add_file(source_fn)
    run.log_artifact(source_cloud_artifact)
    logger.info('Source cloud logged!')
    source_cloud_artifact.wait()

    # Log favorites csv file
    logger.info("Creating favorites csv")
    favorites_csv = wandb.Artifact(
        name=fave_fn,
        type='data',
        description='Csv file of a users favorite Genres and sources')
    favorites_csv.add_file(fave_fn)
    run.log_artifact(favorites_csv)
    logger.info("Favorites data frame logged!")
    favorites_csv.wait()

    save_faves = True
    if save_faves is False:
        os.remove(genre_fn)
        os.remove(source_fn)
        os.remove(fave_fn)


if __name__ == "__main__":
    go()
