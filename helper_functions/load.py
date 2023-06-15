import wandb
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import re
import string
import unicodedata
import ast
import logging

logging.basicConfig(
    filename='./model_recs.log',
    level=logging.INFO,
    filemode='a',
    format='%(asctime)s-%(name)s - %(levelname)s - %(message)s',
    datefmt='%d %b %Y %H:%M:%S %Z',
    force=True)
logger = logging.getLogger()


def get_model(project='anime_recommendations',
              model='wandb_anime_nn.h5:v6',
              artifact_type='h5'):
    run = wandb.init(project=project)
    artifact = run.use_artifact(model, type=artifact_type)
    artifact_path = artifact.file()
    model = tf.keras.models.load_model(artifact_path)
    return model


def get_weights(model):
    anime_weights = model.get_layer('anime_embedding')
    anime_weights = anime_weights.get_weights()[0]
    anime_weights = anime_weights / np.linalg.norm(
        anime_weights, axis=1).reshape((-1, 1))

    user_weights = model.get_layer('user_embedding')
    user_weights = user_weights.get_weights()[0]
    user_weights = user_weights / np.linalg.norm(
        user_weights, axis=1).reshape((-1, 1))
    return anime_weights, user_weights


def get_sypnopses_df(
        project='anime_recommendations',
        sypnopsis_df="sypnopses_artifact_latest",
        artifact_type='raw_data'):
    """
    Download sypnopses df from wandb
    """
    run = wandb.init(project=project)
    artifact = run.use_artifact(sypnopsis_df, type=artifact_type)
    artifact_path = artifact.file()
    cols = ["MAL_ID", "Name", "Genres", "sypnopsis"]
    df = pd.read_csv(artifact_path, usecols=cols)
    return df


def get_anime_df(
        project='anime_recommendations',
        anime_df='all_anime.csv:latest',
        artifact_type='raw_data'):
    """
    Get data frame containing stats on each anime
    """
    run = wandb.init(project=project)  # args.project_name)
    artifact = run.use_artifact(anime_df, type=artifact_type)
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
    run = wandb.init(project=project)
    artifact = run.use_artifact(main_df, type=artifact_type)
    artifact_path = artifact.file()
    df = pd.read_parquet(artifact_path)

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

    return df, anime_to_index, index_to_anime


def main_df_by_id(
        project='anime_recommendations',
        main_df='preprocessed_stats.parquet:v2',
        artifact_type='preprocessed_data'):
    """
    Get data frame from wandb
    Covert to same format we used for neural network
    """
    run = wandb.init(project=project)
    artifact = run.use_artifact(main_df, type=artifact_type)
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
    Get sypnopsis of an anime from the sypnopsis data frame using decoded ID
    """
    if isinstance(anime, int):
        return sypnopsis_df[sypnopsis_df.MAL_ID == anime].sypnopsis.values[0]
    if isinstance(anime, str):
        return sypnopsis_df[sypnopsis_df.Name == anime].sypnopsis.values[0]


def get_genres(anime_df):
    """
    Get all possible anime genres
    Input:
        anime_df: data frame containing all anime, taken from get_anime_df()
    Output:
        genre_list: All possible anime genres in list format
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
    Input:
        anime_df: data frame containing all anime, taken from get_anime_df()
    Output:
        source_list: All possible anime sources in list format
    """
    sources = anime_df['Source'].unique().tolist()
    # Get genres individually (instances have lists of genres)
    possibilities = list(set(str(sources).split()))
    # Remove non alphanumeric characters
    possibilities = sorted(list(
        set([re.sub(r'[\W_]', '', e) for e in possibilities])))
    remove = \
        ['novel', "Light", "Visual", "Picture",
            "Card", "game", "book", "Web", 'nan']
    fixed = possibilities + \
        ['LightNovel', 'VisualNovel', 'PictureBook', 'CardGame', "WebNovel"]
    source_list = sorted([i for i in fixed if i not in remove])
    return source_list


def by_genre(anime_df, genres):
    """
    Restrict the potential anime recommendations according to genre
    Input:
        anime_df: Pandas Data Frame of all anime, taken from get_anime_df()
        genres: List of string genres to include
    Output:
        df: New anime data frame containing only anime of the type(s)
            specified in args.genres
    """
    # Get genres to use and possible genres
    use_genres = ast.literal_eval(genres)
    genres = get_genres(anime_df)
    # Ensure the input genres are valid genres
    for genre in use_genres:
        try:
            assert genre in genres
        except AssertionError:
            return logger.info(
                "An invalid genre was input. Select genres from %s", genres)

    g1, g2, g3 = use_genres
    arr1, arr2, arr3, empty = [], [], [], []

    # Iterate through anime df
    for index, row in anime_df.iterrows():
        i = 0
        # Append an anime to its specific array if it is of the desired genre
        if g1 in str(row['Genres']) and g1 not in arr1[:i] and g1 != "None":
            arr1.append(row)
        if g2 in str(row['Genres']) and g2 not in arr2[:i] and g2 != "None":
            arr2.append(row)
        if g3 in str(row['Genres']) and g3 not in arr3[:i] and g3 != "None":
            arr3.append(row)
        i += 1
    # Initialize empty df
    df = None
    # If array 1 was created, convert to data frame
    if arr1 != empty:
        df = pd.DataFrame(arr1)
    # If array 2 was created, convert to data frame
    if arr2 != empty:
        df2 = pd.DataFrame(arr2)
        # If the first data frame exists, concatenate with it
        if arr1 != empty:
            df = pd.concat([df, df2]).drop_duplicates()
        else:
            df = df2
    # Create third array and concatenate in same manner
    if arr3 != empty:
        df3 = pd.DataFrame(arr3)
        if df is not None:
            df = pd.concat([df, df3]).drop_duplicates()
        else:
            df = df3
    return df


def clean(item):
    """
    Remove or convert all non-alphabetical characters from a string or list
    of strings.
    Strip all Escape characters, accents, spaces, and irregular characters
    """
    translations = []
    irregular = ['★', '♥', '☆', '♡', '½', 'ß', '²']
    if isinstance(item, list):
        for name in item:
            for irr in irregular:
                if irr in name:
                    name = name.replace(irr, ' ')
            x = str(name).translate({ord(c): None for c in string.whitespace})
            x = re.sub(r'\W+', '', x)
            x = u"".join([c for c in unicodedata.normalize('NFKD', x)
                          if not unicodedata.combining(c)])

            translations.append(x.lower())
    else:
        for irr in irregular:
            if irr in item:
                item = item.replace(irr, ' ')
        x = str(item).translate({ord(c): None for c in string.whitespace})
        x = re.sub(r'\W+', '', x)
        x = u"".join([c for c in unicodedata.normalize('NFKD', x)
                      if not unicodedata.combining(c)])
        return x.lower()

    return translations
