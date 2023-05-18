import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# import sys
# from pathlib import Path
# path_root = Path(__file__.parent[1])
# sys.path.append(str(path_root))
from helper_functions.load import get_genres, get_sources


def get_random_user(rating_df, user_to_index, index_to_user):
    """
    Get a random user from main data frame
    """
    possible_users = list(user_to_index.keys())

    random_user = int(random.choice(possible_users))
    return random_user


def genre_cloud(
        anime_df,
        ID,
        width=800,
        height=600,
        genre_fn=None,
        show_clouds=False,
        interval=3000):
    genres = get_genres(anime_df)
    genres = (" ").join(list(map(str.upper, genres)))

    cloud = WordCloud(width=width,
                      height=height,
                      prefer_horizontal=0.85,
                      background_color='white',
                      contour_width=0.05,
                      colormap='spring')
    genres_cloud = cloud.generate(genres)
    if genre_fn is not None:
        fn = "User_ID_" + str(ID) + '_' + str(genre_fn)
        genres_cloud.to_file(fn)
    if show_clouds is True:
        show_cloud(genres_cloud, interval=interval)
    return genres_cloud, fn


def source_cloud(
        anime_df,
        ID,
        width=800,
        height=600,
        sources_fn=None,
        show_clouds=False,
        interval=3000):
    source = get_sources(anime_df)
    sources = (" ").join(list(map(str.upper, source)))

    cloud = WordCloud(width=width,
                      height=height,
                      prefer_horizontal=0.85,
                      background_color='white',
                      contour_width=0.05,
                      colormap='spring')
    source_cloud = cloud.generate(sources)
    if sources_fn is not None:
        fn = 'User_ID_' + str(ID) + '_' + str(sources_fn)
        source_cloud.to_file(fn)
    if show_clouds is True:
        show_cloud(source_cloud, interval=interval)
    return source_cloud, fn


def show_cloud(cloud, interval=3000):
    fig = plt.figure(figsize=(8, 6))
    timer = fig.canvas.new_timer(interval=interval)
    timer.add_callback(plt.close)
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    timer.start()
    plt.show()


def fave_genres(user, df, anime_df, favorite_percentile=80):
    watched = df[df.user_id == user]
    user_rating_percentile = np.percentile(
        watched.rating, favorite_percentile)
    watched = watched[watched.rating >= user_rating_percentile]
    top = (watched.sort_values(by="rating", ascending=False).anime_id.values)

    faves = anime_df[anime_df["anime_id"].isin(top)]
    faves = faves[["eng_version", "Genres"]]
    return pd.DataFrame(faves)


def fave_sources(user, df, anime_df, favorite_percentile=0.8):
    watched = df[df.user_id == user]
    user_rating_percentile = np.percentile(
        watched.rating, favorite_percentile)
    watched = watched[watched.rating >= user_rating_percentile]
    top = (watched.sort_values(by="rating", ascending=False).anime_id.values)

    faves = anime_df[anime_df["anime_id"].isin(top)]
    faves = faves[["eng_version", "Source"]]
    return pd.DataFrame(faves)
