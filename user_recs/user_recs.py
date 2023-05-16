import string
import argparse
import logging
import os
import wandb
import pandas as pd
from distutils.util import strtobool
import re
import random
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import tensorflow as tf

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


def find_similar_users(user_id, n_users):
    rating_df, user_to_index, index_to_user = get_main_df()
    anime_weights, user_weights = get_weights()
    weights = user_weights

    try:
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
                SimilarityArr.append(
                    {"similar_users": decoded_id,
                     "similarity": similarity})

        Frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity",
                                                        ascending=False)

        return Frame

    except BaseException:
        logger.info('%s Not Found in User list', user_id)


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


def get_anime_frame(anime, df):
    """
    Get either the anime's name or id as a data frame
    """
    if isinstance(anime, int):
        return df[df.anime_id == anime]
    if isinstance(anime, str):
        return df[df.eng_version == anime]


def get_sypnopses_df():
    """
    Download sypnopses df from wandb
    """
    run = wandb.init(project=args.project_name)
    logger.info("Downloading sypnopses df")
    artifact = run.use_artifact(args.sypnopses_df, type='raw_data')
    artifact_path = artifact.file()
    cols = ["MAL_ID", "Name", "Genres", "sypnopsis"]
    df = pd.read_csv(artifact_path, usecols=cols)
    logger.info("Anime df shape is %s", df.shape)
    return df


def get_sypnopsis(anime, sypnopsis_df):
    """
    Get sypnopsis of an anime from the sypnopsis data frame
    """
    if isinstance(anime, int):
        return sypnopsis_df[sypnopsis_df.MAL_ID == anime].sypnopsis.values[0]
    if isinstance(anime, str):
        return sypnopsis_df[sypnopsis_df.Name == anime].sypnopsis.values[0]


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
    logger.info("Orignal anime df shape is %s", df.shape)
    df = df.replace("Unknown", np.nan)

    df['anime_id'] = df['MAL_ID']
    df['japanese_name'] = df['Japanese name']
    df["eng_version"] = df['English name']
    logger.info("Original English version is %s", df["eng_version"].head())
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
    logger.info("Final anime df shape is %s", df.shape)
    return df


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
    # index_to_anime = {count: value for count, value in enumerate(anime_ids)}

    # Convert values of format id to count_number
    df["user"] = df["user_id"].map(user_to_index)
    df["anime"] = df["anime_id"].map(anime_to_index)
    df = df[['user', 'anime', 'rating', 'user_id', 'anime_id']]
    df = df.sample(frac=1, random_state=42)

    # logger.info("Final preprocessed df shape is %s", df.shape)
    # logger.info("Final preprocessed df head is %s", df.head())

    return df, user_to_index, index_to_user


def get_random_user():
    """
    Get a random user from main data frame
    """
    rating_df, user_to_index, index_to_user = get_main_df()
    possible_users = list(user_to_index.keys())

    random_user = int(random.choice(possible_users))
    return random_user


def fave_genres(user, df, anime_df):
    # Instances where input user is in data frame's user_id column
    watched = df[df.user_id == user]
    logger.info("watched head looks like %s", watched.head())
    user_rating_percentile = np.percentile(
        watched.rating, float(args.favorite_percentile))
    watched = watched[watched.rating >= user_rating_percentile]
    top = (watched.sort_values(by="rating", ascending=False).anime_id.values)

    faves = anime_df[anime_df["anime_id"].isin(top)]
    faves = faves[["eng_version", "Genres"]]
    return pd.DataFrame(faves)


def fave_sources(user, df, anime_df):
    watched = df[df.user_id == user]
    user_rating_percentile = np.percentile(
        watched.rating, float(args.favorite_percentile))
    watched = watched[watched.rating >= user_rating_percentile]
    top = (watched.sort_values(by="rating", ascending=False).anime_id.values)

    faves = anime_df[anime_df["anime_id"].isin(top)]
    faves = faves[["eng_version", "Source"]]
    return pd.DataFrame(faves)


def fave_df(user, df, anime_df):

    def fave_sources():
        watched = df[df.user_id == user]
        user_rating_percentile = np.percentile(watched.rating, 75)
        watched = watched[watched.rating >= user_rating_percentile]
        top = (
            watched.sort_values(
                by="rating",
                ascending=False).anime_id.values)
        faves = anime_df[anime_df["anime_id"].isin(top)]
        faves = faves[["eng_version", "Source"]]
        return pd.DataFrame(faves)

    def fave_genres():
        # df, user_to_index, index_to_user = get_main_df2()

        # anime_df = get_anime_df()
        watched = df[df.user_id == user]
        user_rating_percentile = np.percentile(watched.rating, 75)
        watched = watched[watched.rating >= user_rating_percentile]
        top = (
            watched.sort_values(
                by="rating",
                ascending=False).anime_id.values)

        faves = anime_df[anime_df["anime_id"].isin(top)]
        faves = faves[["eng_version", "Genres"]]

    # print("> User #{} has rated {} anime (avg. rating = {:.1f})".format(
    #      user, len(watched),
    #      watched['rating'].mean(),
    #    ))

    # print('> preferred genres')

    # genre_cloud(faves)
        return pd.DataFrame(faves)
    genres = fave_genres()
    sources = fave_sources()
    genres = genres["Genres"]
    sources["Genres"] = genres
    return sources


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
    fn = "User_ID_" + str(ID) + '_' + args.genre_fn
    genres_cloud.to_file(fn)
    if args.show_clouds is True:
        show_cloud(genre_cloud)
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
    fn = 'User_ID_' + str(ID) + '_' + args.source_fn
    source_cloud.to_file(fn)
    if args.show_clouds is True:
        show_cloud(source_cloud)
    return source_cloud, fn


def show_cloud(cloud):
    fig = plt.figure(figsize=(8, 6))
    timer = fig.canvas.new_timer(interval=int(args.interval))
    timer.add_callback(plt.close)
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    timer.start()
    plt.show()


def similar_user_recs(user, n=10):
    # Find anime that similar users have seen but input user has not
    # The more commonly seen those anime were, the higher they are
    # Prioritized
    # N is the number of similar users to compare with
    sypnopsis_df = get_sypnopses_df()
    rating_df, user_to_index, index_to_user = get_main_df()
    anime_df = get_anime_df()
    # genre_df = fave_genres(user, rating_df, anime_df)
    # source_df = fave_sources(user, rating_df, anime_df)
    similar_users = find_similar_users(user, n_users=n)
    # get_fave_df(genre_df, source_df, user)
    user_pref = fave_df(user, rating_df, anime_df)

    # user_pref['eng'] = clean(user_pref.eng_version.values.tolist())
    recommended_animes, anime_list = [], []
    eng_versions = user_pref.eng_version.values.tolist()
    # clean_eng = clean(eng_versions)
    # Get list of similar user IDs
    for user_id in similar_users.similar_users.values:
        # genre_df = fave_genres(user_id, rating_df, anime_df)
        # source_df = fave_sources(user_id, rating_df, anime_df)
        # pref_list = get_fave_df(genre_df, source_df, user_id)
        pref_list = fave_df(user_id, rating_df, anime_df)
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
    df.to_csv(filename, index=False)
    return df, filename


def go(args):
    # Initialize run
    run = wandb.init(
        project=args.project_name,
        name="user_genre_based_preferences_recommendations")

    if args.use_random_user is True:
        user = get_random_user()
        logger.info("Using %s as random input user", user)
    else:
        user = int(args.user_query)
        logger.info("Using %s as input user", user)

    # Create data frame file
    _, filename = similar_user_recs(user, n=int(args.user_num_recs))
    df, user_to_index, index_to_user = get_main_df()
    anime_df = get_anime_df()

    genre_df = fave_genres(user, df, anime_df)
    source_df = fave_sources(user, df, anime_df)

    genres_cloud, genre_fn = genre_cloud(genre_df, user)
    sources_cloud, source_fn = source_cloud(source_df, user)
    fave_df, fave_fn = get_fave_df(
        genre_df, source_df, user, save=args.save_faves)

    # Create artifact
    logger.info("Creating artifact")
    description = "Anime recs based on user prefs: " + str(user)
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
