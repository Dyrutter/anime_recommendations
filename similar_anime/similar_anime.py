import argparse
import logging
import wandb
import string
import os
import pandas as pd
import numpy as np
from distutils.util import strtobool
import unicodedata
import random
import ast
import re
import tensorflow as tf

logging.basicConfig(
    filename='./similar_anime.log',  # Path to log file
    level=logging.INFO,  # Log info, warnings, errors, and critical errors
    filemode='a',  # Create log file if one doesn't already exist and add
    format='%(asctime)s-%(name)s - %(levelname)s - %(message)s',
    datefmt='%d %b %Y %H:%M:%S %Z',  # Format date
    force=True)
logger = logging.getLogger()


def main_df_by_anime():
    """
    Get main df from wandb and covert to same format used for neural network
    Outputs:
        df: Main data frame with added columns containing additional indices
        anime_to_index: enumerated anime IDs of format {ID: enumerated_index}
        index_to_anime: enumerated anime IDs of format {enumerated_index: ID}
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
    index_to_anime = {count: value for count, value in enumerate(anime_ids)}

    # Convert values of format id to count_number
    df["user"] = df["user_id"].map(user_to_index)
    df["anime"] = df["anime_id"].map(anime_to_index)
    df = df[['user', 'anime', 'rating', 'user_id', 'anime_id']]
    df = df.sample(frac=1, random_state=42)

    return df, anime_to_index, index_to_anime


def get_anime_df():
    """
    Get data frame containing stats on each anime from wandb.
    Create column of cleaned anime names for filename usage.
    """
    run = wandb.init(project=args.project_name)
    artifact = run.use_artifact(args.anime_df, type=args.anime_df_type)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path)
    df = df.replace("Unknown", np.nan)

    df['anime_id'] = df['MAL_ID']
    df['japanese_name'] = df['Japanese name']
    df["eng_version"] = df['English name']
    # df['eng_version'] = df.anime_id.apply(lambda x: get_anime_name(x, df))
    df['full_eng_version'] = df["Name"]

    # Get column of cleaned anime names
    df['eng_version'] = df.anime_id.apply(
        lambda x: clean(get_anime_name(x, df)).lower())
    df.sort_values(by=['Score'],
                   inplace=True,
                   ascending=False,
                   kind='quicksort',
                   na_position='last')
    keep_cols = ["anime_id", "eng_version", "Score", "Genres", "Episodes",
                 "Premiered", "Studios", "japanese_name", "Name", "Type",
                 "Source", 'Rating', 'Members', 'full_eng_version']
    df = df[keep_cols]
    return df


def get_anime_name(anime_id, df):
    """
    Helper function for loading anime data frame
    Inputs:
        anime_id: The ID of an anime
        df: anime stats data frame
    Outputs:
        name: The english name of anime_id
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


def get_sypnopses_df():
    """
    Download sypnopses df from wandb
    """
    run = wandb.init(project=args.project_name)
    art = run.use_artifact(args.sypnopses_df, type=args.sypnopsis_df_type)
    artifact_path = art.file()
    cols = ["MAL_ID", "Name", "Genres", "sypnopsis"]
    df = pd.read_csv(artifact_path, usecols=cols)
    return df


def get_model():
    """
    Download neural network model from wandb
    """
    run = wandb.init(project=args.project_name)
    artifact = run.use_artifact(args.model, type=args.model_type)
    artifact_path = artifact.file()
    model = tf.keras.models.load_model(artifact_path)
    return model


def get_weights(model):
    """
    Extract weights from model and apply Frobenius normalization.
    Inputs:
        model: neural network model
    Outputs:
        anime_weights: norm weights associated with anime embedding layer
        user_weights: norm weights associated with user embedding layer
    """
    anime_weights = model.get_layer(args.anime_emb_name)
    anime_weights = anime_weights.get_weights()[0]
    anime_weights = anime_weights / np.linalg.norm(
        anime_weights, axis=1).reshape((-1, 1))

    user_weights = model.get_layer(args.ID_emb_name)
    user_weights = user_weights.get_weights()[0]
    user_weights = user_weights / np.linalg.norm(
        user_weights, axis=1).reshape((-1, 1))
    return anime_weights, user_weights


def get_genres(anime_df):
    """
    Get all possible anime genres
    Input: data frame containing anime statistics
    Output: All possible anime genres in list format
    """
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


def get_sypnopsis(anime, sypnopsis_df):
    """
    Get sypnopsis of an anime from the sypnopsis data frame using decoded ID
    """
    if isinstance(anime, int):
        return sypnopsis_df[sypnopsis_df.MAL_ID == anime].sypnopsis.values[0]
    if isinstance(anime, str):
        return sypnopsis_df[sypnopsis_df.Name == anime].sypnopsis.values[0]


def get_anime_frame(anime, df, clean=False):
    """
    Get either the anime df containing only specified anime
    """
    if isinstance(anime, int):
        return df[df.anime_id == anime]
    if isinstance(anime, str):
        if clean is False:
            return df[df.Name == anime]
        else:
            return df[df.eng_version == anime]


def get_random_anime(anime_df):
    """
    Get a random anime from anime data frame
    """
    possible_anime = anime_df['Name'].unique().tolist()  # was eng_version
    random_anime = random.choice(possible_anime)
    return random_anime


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


def by_genre(anime_df):
    """
    Restrict the potential anime recommendations according to genre
    """
    # Get genres to use and possible genres
    use_genres = ast.literal_eval(args.genres)
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


def get_types(df):
    """
    Modify data frame to include only anime of specified genre
    """
    possibilities = ['TV', 'OVA', 'Movie', 'Special', 'ONA', 'Music']
    use_types = ast.literal_eval(args.types)
    for anime_type in use_types:
        try:
            assert anime_type in possibilities
        except AssertionError:
            return logger.info(
                "An invalid type was input. Select from %s", possibilities)
    return use_types


pd.set_option("max_colwidth", None)


def anime_recs(name, count, anime_df):
    """
    Get anime recommendations based on similar anime.
    Count is the number of similar anime to return based on highest score
    """
    sypnopsis_df = get_sypnopses_df()
    model = get_model()
    weights, _ = get_weights(model)
    rating_df, anime_to_index, index_to_anime = main_df_by_anime()
    use_types = get_types(anime_df)

    translated = clean(name)
    filename = translated + '.csv'
    logger.info('filename is %s', filename)

    # Get ID and encoded index of input anime
    try:
        index = get_anime_frame(translated, anime_df).anime_id.values[0]
    except IndexError:
        try:
            # In case the name has special characters
            index = get_anime_frame(name, anime_df).anime_id.values[0]
        except IndexError:
            # In case there is a typo in the config file
            index = get_anime_frame(
                translated, anime_df, clean=True).anime_id.values[0]

    logger.info('index is %s', index)
    encoded_index = anime_to_index.get(index)

    # Get and sort dists
    dists = np.dot(weights, weights[encoded_index])
    sorted_dists = np.argsort(dists)
    closest = sorted_dists[:]
    arr = []

    # Sequentially append closest animes to empty array
    for close in closest:
        decoded_id = index_to_anime.get(close)
        anime_frame = get_anime_frame(decoded_id, anime_df)

        # Some anime do not have sypnopses
        try:
            sypnopsis = get_sypnopsis(decoded_id, sypnopsis_df)
        except BaseException:
            sypnopsis = "None"

        # Get desired column values for anime
        full = anime_frame["Name"].values[0]
        genre = anime_frame['Genres'].values[0]
        japanese_name = anime_frame['japanese_name'].values[0]
        episodes = anime_frame['Episodes'].values[0]
        premiered = anime_frame['Premiered'].values[0]
        studios = anime_frame['Studios'].values[0]
        score = anime_frame["Score"].values[0]
        Type = anime_frame['Type'].values[0]
        source = anime_frame['Source'].values[0]
        rating = anime_frame["Rating"].values[0]
        similarity = dists[close]

        # Don't include anime if they aren't of a specified type
        if args.spec_types is True:
            if Type in use_types:
                arr.append(
                    {"anime_id": decoded_id, "Name": full,
                     "Similarity": similarity, "Genres": genre,
                     'Sypnopsis': sypnopsis, "Episodes": episodes,
                     "Japanese name": japanese_name, "Studios": studios,
                     "Premiered": premiered, "Score": score,
                     "Type": Type, "Source": source, 'Rating': rating})
        else:
            arr.append(
                {"anime_id": decoded_id, "Name": full,
                 "Similarity": similarity, "Genres": genre,
                 'Sypnopsis': sypnopsis, "Episodes": episodes,
                 "Japanese name": japanese_name, "Studios": studios,
                 "Premiered": premiered, "Score": score,
                 "Type": Type, "Source": source, 'Rating': rating})

    # Convert array to data frame, ensuring the input anime is not included
    Frame = pd.DataFrame(arr)
    Frame = Frame[Frame.anime_id != index].drop(['anime_id'], axis=1)

    # Remove anime not of specified genre if so desired
    if args.spec_genres is True:
        Frame = by_genre(Frame).sort_values(
            by="Similarity", ascending=False).drop(['Genres'], axis=1)
    else:
        Frame = Frame.sort_values(by="Similarity", ascending=False)
    return Frame[:count], filename, translated


def go(args):
    # Initialize run
    run = wandb.init(project=args.project_name)
    anime_df = get_anime_df()

    if args.random_anime is True:
        anime_name = get_random_anime(anime_df)
        logger.info("Using %s as random input anime", anime_name)
    else:
        anime_name = args.anime_query

    # Create data frame file
    df, fn, name = anime_recs(anime_name, int(args.a_query_number), anime_df)
    df.to_csv(fn, index=False)

    # Create artifact
    logger.info("Creating recommendations artifact")
    description = "Anime most similar to: " + str(anime_name)
    recs_artifact = wandb.Artifact(
        name=fn,
        type=args.a_rec_type,
        description=description,
        metadata={"Queried anime: ": anime_name,
                  "Model used: ": args.model,
                  "Main data frame used: ": args.main_df,
                  "Filename: ": fn})

    # Upload artifact to wandb
    recs_artifact.add_file(fn)
    logger.info("Logging artifact for anime %s", anime_name)
    run.log_artifact(recs_artifact)
    recs_artifact.wait()

    if args.save_sim_anime is False:
        os.remove(fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get recommendations based on similar anime",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--main_df_type",
        type=str,
        help="Artifact type of main data frame",
        required=True
    )

    parser.add_argument(
        "--anime_df_type",
        type=str,
        help="Artifact type of anime df",
        required=True
    )

    parser.add_argument(
        "--sypnopsis_df_type",
        type=str,
        help='Artifact type of sypnopses df',
        required=True
    )

    parser.add_argument(
        "--model_type",
        type=str,
        help='Artifact type of neural network',
        required=True
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Wandb artifact with .h5 file of neural network",
        required=True
    )

    parser.add_argument(
        "--project_name",
        type=str,
        help="Name of wandb project",
        required=True
    )

    parser.add_argument(
        "--main_df",
        type=str,
        help="Main prerpocessed data frame",
        required=True
    )

    parser.add_argument(
        "--sypnopses_df",
        type=str,
        help="Sypnopses df",
        required=True
    )

    parser.add_argument(
        "--anime_df",
        type=str,
        help="anime df",
        required=True
    )

    parser.add_argument(
        "--anime_query",
        type=str,
        help="list of input names of anime to query",
        required=True
    )

    parser.add_argument(
        "--a_query_number",
        type=str,
        help="Number of similar anime to return",
        required=True
    )

    parser.add_argument(
        "--random_anime",
        type=lambda x: bool(strtobool(x)),
        help="Whether to use a random anime",
        required=True
    )

    parser.add_argument(
        "--genres",
        type=str,
        help="List of genres to narrow down return values",
        required=True
    )

    parser.add_argument(
        "--spec_genres",
        type=lambda x: bool(strtobool(x)),
        help="Boolean of whether or not to narrow down by specific genres",
        required=True
    )

    parser.add_argument(
        "--types",
        type=str,
        help="List of types of anime to return ['TV', 'OVA', etc]",
        required=True
    )

    parser.add_argument(
        "--spec_types",
        type=lambda x: bool(strtobool(x)),
        help="Boolean of whether or not to specify types of anime to return",
        required=True
    )

    parser.add_argument(
        "--a_rec_type",
        type=str,
        help="Type of artifact anime recommendations are saved as",
        required=True
    )

    parser.add_argument(
        "--save_sim_anime",
        type=lambda x: bool(strtobool(x)),
        help="Boolean of whether to save anime recs to local machine",
        required=True
    )

    parser.add_argument(
        "--ID_emb_name",
        type=str,
        help="Name of user weight layer in neural network model",
        required=True
    )

    parser.add_argument(
        "--anime_emb_name",
        type=str,
        help="Name of anime weight layer in neural network model",
        required=True
    )

    args = parser.parse_args()
    go(args)
