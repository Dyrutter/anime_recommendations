from helper_functions.load import get_anime_frame, get_sypnopsis
from helper_functions.load import get_weights, get_sypnopses_df
from helper_functions.load import main_df_by_id, get_anime_df, get_model
from helper_functions.User import get_random_user, genre_cloud, source_cloud
from helper_functions.User import fave_genres, fave_sources
import string
import argparse
import logging
import os
import wandb
import pandas as pd
from distutils.util import strtobool
import re
import numpy as np
import sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))


logging.basicConfig(
    filename='./user_recs.log',
    level=logging.INFO,
    filemode='a',
    format='%(asctime)s-%(name)s - %(levelname)s - %(message)s',
    datefmt='%d %b %Y %H:%M:%S %Z',
    force=True)
logger = logging.getLogger()


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


def find_similar_users(user_id, n_users, rating_df, user_to_index,
                       index_to_user, user_weights):

    try:
        index = user_id
        encoded_index = user_to_index.get(index)

        dists = np.dot(user_weights, user_weights[encoded_index])
        sorted_dists = np.argsort(dists)
        n_users = n_users + 1
        closest = sorted_dists[-n_users:]

        SimilarityArr = []

        for close in closest:
            similarity = dists[close]
            decoded_id = index_to_user.get(close)
            if decoded_id != user_id:
                SimilarityArr.append(
                    {"similar_users": decoded_id,
                     "similarity": similarity})

        Frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity",
                                                        ascending=False)

        return Frame

    except BaseException:
        logger.info('%s Not Found in User list', user_id)


def get_fave_df(genres, sources, ID, save=False):
    """
    Input source and genre dfs and returned merged df
    """
    genres = genres["Genres"]
    sources["Genres"] = genres
    if save is True:
        fn = 'User_ID_' + str(ID) + '_' + args.prefs_csv
        sources.to_csv(fn)
        return sources, fn
    else:
        return sources


def similar_user_recs(
        user,
        similar_users,
        sypnopsis_df,
        rating_df,
        user_to_index,
        index_to_user,
        anime_df,
        n,
        genre_df,
        source_df):
    # Find anime that similar users have seen but input user has not
    # The more commonly seen those anime were, the higher they are
    # Prioritized
    user_pref = get_fave_df(genre_df, source_df, user)

    # user_pref['eng'] = clean(user_pref.eng_version.values.tolist())
    recommended_animes, anime_list = [], []
    eng_versions = user_pref.eng_version.values.tolist()
    # clean_eng = clean(eng_versions)
    # Get list of similar user IDs
    for user_id in similar_users.similar_users.values:
        genre_df = fave_genres(user_id, rating_df, anime_df)
        source_df = fave_sources(user_id, rating_df, anime_df)
        # pref_list = get_fave_df(genre_df, source_df, user_id)
        pref_list = get_fave_df(genre_df, source_df, user)
        # pref_list['eng'] = clean(pref_list.eng_version.values.tolist())
        # Favorites of similar users that input user has not watched
        pref_list = pref_list[~pref_list.eng_version.isin(eng_versions)]
        anime_list.append(pref_list.eng_version.values)

    anime_list = pd.DataFrame(anime_list)
    sorted_list = pd.DataFrame(
        pd.Series(anime_list.values.ravel()).value_counts()).head(n)

    for i, anime_name in enumerate(sorted_list.index):
        n_pref = sorted_list[sorted_list.index == anime_name].values[0][0]
        anime_frame = get_anime_frame(anime_name, anime_df)
        name = anime_frame['eng_version'].values[0]
        genre = anime_frame['Genres'].values[0]
        japanese_name = anime_frame['japanese_name'].values[0]
        episodes = anime_frame['Episodes'].values[0]
        premiered = anime_frame['Premiered'].values[0]
        studios = anime_frame['Studios'].values[0]
        score = anime_frame["Score"].values[0]
        Type = anime_frame['Type'].values[0]
        source = anime_frame['Source'].values[0]
        anime_id = anime_frame['anime_id'].values[0]
        sypnopsis = get_sypnopsis(int(anime_id), sypnopsis_df)
        recommended_animes.append(
            {"anime_id": anime_id, "Name": name, "n_user_prefs": n_pref,
             "Source": source, "Genres": genre,
             'Sypnopsis': sypnopsis, "Episodes": episodes,
             "Japanese name": japanese_name, "Studios": studios,
             "Premiered": premiered, "Score": score, "Type": Type})
    filename = 'User_ID_' + str(user) + '_' + args.user_recs_fn
    df = pd.DataFrame(recommended_animes)
    # df.to_csv(filename, index=False)
    return df, filename


def go(args):
    # Initialize run
    run = wandb.init(
        project=args.project_name,
        name="user_genre_based_preferences_recommendations")

    df, user_to_index, index_to_user = main_df_by_id(
        project=args.project_name,
        main_df=args.main_df,
        artifact_type='preprocessed_data')

    anime_df = get_anime_df(
        project=args.project_name,
        anime_df=args.anime_df,
        artifact_type='raw_data')
    sypnopsis_df = get_sypnopses_df(
        project=args.project_name,
        sypnopsis_df=args.sypnopses_df,
        artifact_type='raw_data')
    model = get_model(
        project=args.project_name,
        model=args.model,
        artifact_type='h5')
    anime_weights, user_weights = get_weights(model)

    if args.use_random_user is True:
        user = get_random_user()
        logger.info("Using %s as random input user", user)
    else:
        user = int(args.user_query)
        logger.info("Using %s as input user", user)

    genre_df = fave_genres(user, df, anime_df,
                           favorite_percentile=int(args.favorite_percentile))

    source_df = fave_sources(
        user, df, anime_df, favorite_percentile=int(
            args.favorite_percentile))

    similar_users = find_similar_users(user,
                                       int(args.user_num_recs),
                                       df,
                                       user_to_index,
                                       index_to_user,
                                       user_weights)
    similar_users_df, filename = similar_user_recs(
        user, similar_users, sypnopsis_df, df, user_to_index,
        index_to_user, anime_df, int(args.user_num_recs), genre_df, source_df)
    similar_users_df.to_csv(filename, index=False)

    genres_cloud, genre_fn = genre_cloud(
        genre_df, user, width=int(args.cloud_width),
        genre_fn=args.genre_fn, show_clouds=args.show_clouds,
        height=int(args.cloud_height), interval=int(args.interval))
    sources_cloud, source_fn = source_cloud(
        source_df, user, width=int(args.cloud_width),
        height=int(args.cloud_height), sources_fn=args.source_fn,
        show_clouds=args.show_clouds, interval=int(args.interval))
    fave_df, fave_fn = get_fave_df(
        genre_df, source_df, user, save=args.save_faves)

    # Create artifact
    logger.info("Creating similar user based recs artifact")
    description = "Anime recs based on user prefs: " + str(user)
    artifact = wandb.Artifact(
        name=filename,
        type="csv",
        description=description,
        metadata={"Queried user: ": user})
    artifact.add_file(filename)
    logger.info("Logging artifact for user %s", user)
    run.log_artifact(artifact)
    artifact.wait()

    # Log favorite genre cloud
    logger.info("Genre Cloud artifact")
    genre_cloud_artifact = wandb.Artifact(
        name=genre_fn,
        type="image",
        description='Cloud image of favorite genres',
        metadata={'Queried user: ': user})
    genre_cloud_artifact.add_file(genre_fn)
    run.log_artifact(genre_cloud_artifact)
    logger.info("Genre cloud logged!")
    genre_cloud_artifact.wait()

    # Log favorite source cloud
    logger.info("Creating source cloud artifact")
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
    if args.save_user_recs is False:
        os.remove(filename)
        os.remove(source_fn)
        os.remove(fave_fn)
        os.remove(genre_fn)


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
        "--use_random_user",
        type=lambda x: bool(strtobool(x)),
        help="Decide whether or not to use a random user id",
        required=True
    )

    parser.add_argument(
        "--user_recs_fn",
        type=str,
        help="Artifact name of user recommendations csv file",
        required=True
    )

    parser.add_argument(
        "--save_user_recs",
        type=lambda x: bool(strtobool(x)),
        help="Whether or not to save user recs file locally",
        required=True
    )

    parser.add_argument(
        "--sypnopses_df",
        type=str,
        help="Sypnopses df",
        required=True
    )
    parser.add_argument(
        "--user_num_recs",
        type=str,
        help="Number of anime recommendations to return",
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
        "--model",
        type=str,
        help="Wandb artifact with .h5 file of neural network",
        required=True
    )
    args = parser.parse_args()
    go(args)
