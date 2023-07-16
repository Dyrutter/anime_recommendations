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
    Load data frame artifact containing info on each anime from wandb.
    Create column of cleaned anime names for filename usage.
    Output:
        df: Pandas Data Frame containing all anime and their relevant stats
    """
    run = wandb.init(project=args.project_name)
    artifact = run.use_artifact(args.anime_df, type=args.anime_df_type)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path)
    df = df.replace("Unknown", np.nan)

    df['anime_id'] = df['MAL_ID']
    df['japanese_name'] = df['Japanese name']
    df["eng_version"] = df['English name']

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
                 "Source", 'Rating', 'Members']
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
    name = df[df.anime_id == anime_id].Name.values[0]
    return name


def get_sypnopses_df():
    """
    Download sypnopses data frame artifact from wandb
    Output:
        df: Pandas Data Frame containting columns
            ["MAL_ID", "Name", "Genres", "sypnopsis"]
    """
    run = wandb.init(project=args.project_name)
    art = run.use_artifact(args.sypnopses_df, type=args.sypnopsis_df_type)
    artifact_path = art.file()
    cols = ["MAL_ID", "Name", "Genres", "sypnopsis"]
    df = pd.read_csv(artifact_path, usecols=cols)
    return df


def get_model():
    """
    Download neural network model artifact from wandb
    Output:
        model: TensorFlow neural network model
    """
    run = wandb.init(project=args.project_name)
    artifact = run.use_artifact(args.model, type=args.model_type)
    artifact_path = artifact.file()
    model = tf.keras.models.load_model(artifact_path)
    return model


def get_weights(model):
    """
    Extract model weights and apply Frobenius normalization. Normalization
    accounts for magnitude differences in embedding/feature vectors,
    ensuring users/anime with large embeddings don't dominate and instead
    reflect the relative relationships.

    Later, the dot product of either weight matrix can be taken with the
    transposed version of itself (e.g. anime_weights•anime_weightsT) to find
    similarities between animes, as each element in the dot product matrix is
    normalized by the product of the corresponding magnitudes of the anime's
    embeddings/characteristics.
    Inputs:
        model: neural network model
    Outputs:
        anime_weights: normalized weights of anime embedding layer
        user_weights: normalized weights of user embedding layer
    """
    # Get anime weights layer, name specified in config file
    anime_weights = model.get_layer(args.anime_emb_name)
    # Shape of get_weights()[0] is (17560, 128) AKA (num anime, embedding dim)
    anime_weights = anime_weights.get_weights()[0]
    # Normalized embedding vectors (1 value) for each anime, shape (17560, 1)
    anime_norm = np.linalg.norm(anime_weights, axis=1).reshape((-1, 1))
    # Divide anime weights by normalized embedding vector value for each anime
    anime_weights = anime_weights / anime_norm

    # Get user weights layer, name specified in config file
    user_weights = model.get_layer(args.ID_emb_name)
    # Shape of get_weights()[0] is (91641, 128), AKA (num users, emb dim)
    user_weights = user_weights.get_weights()[0]
    # Normalized embedding vectors (1 value) for each user, shape (91641, 1)
    user_norm = np.linalg.norm(user_weights, axis=1).reshape((-1, 1))
    # Divide user weights by normalized embedding vector for each user
    user_weights = user_weights / user_norm
    return anime_weights, user_weights


def get_genres(anime_df):
    """
    Get all possible anime genres
    Input: data frame containing anime statistics taken from get_anime_df()
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
    Helper function to get the sypnopsis of an anime in data frame format
    Input:
        anime: Either the string name of the anime or an int of the anime's ID
        sypnopsis_df: Data Frame containing list of all anime
    Output:
        sypnopsis_df: An anime's sypnopsis in Data Frame format
    """
    if isinstance(anime, int):
        return sypnopsis_df[sypnopsis_df.MAL_ID == anime].sypnopsis.values[0]
    if isinstance(anime, str):
        return sypnopsis_df[sypnopsis_df.Name == anime].sypnopsis.values[0]


def get_anime_frame(anime, df, clean=False):
    """
    Helper function to get a specific anime in data frame format
    Input:
        anime: Either the string name of the anime or an int of the anime's ID
        df: Data Frame containing list of all anime taken from get_anime_df()
        clean: If True, return the name of the anime cleaned with clean()
           If False, return the anime's full name
    Output:
        df: Pandas data frame of stats for a specific anime
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
    Input:
        anime_df: Data frame containing all anime, taken from get_anime_df()
    Output:
        str, a random anime
    """
    possible_anime = anime_df['Name'].unique().tolist()
    random_anime = random.choice(possible_anime)
    return random_anime


def clean(item):
    """
    Remove or convert all non-alphabetical characters from a string or list
    of strings.
    Strip all Escape characters, accents, spaces, and irregular characters
    Inputs:
        item: either a list of string or a string
    Outputs:
        translations: a list of cleaned strings if item was a list, or
            a cleaned string if item was a string
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
    Input:
        anime_df: Pandas Data Frame of all anime, taken from get_anime_df()
    Output:
        df: New anime data frame containing only anime of the type(s)
            specified in args.anime_rec_genres
    """
    # Get genres to use and possible genres
    use_genres = clean(ast.literal_eval(args.anime_rec_genres))
    genres = clean(get_genres(anime_df))

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
        if g1 in str(row['Genres']).lower().replace(
                " ", "") and g1 not in arr1[:i] and g1 != "none":
            arr1.append(row)

        if g2 in str(row['Genres']).lower().replace(
                " ", "") and g2 not in arr2[:i] and g2 != "none":
            arr2.append(row)

        if g3 in str(row['Genres']).lower().replace(
                " ", "") and g3 not in arr3[:i] and g3 != "none":
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


def get_types():
    """
    Confirm list of types input in args.types from the list
        ['TV', 'OVA', 'Movie', 'Special', 'ONA', 'Music']
    Output:
        list, types of anime to include in recommendations
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
    Inputs:
        name: Str, name of the anime to find similar recommendations for
        count: Int, number of recommendations to include
        anime_df: Pandas Data Frame of all anime, taken from get_anime_df()
    Outputs:
        Frame: Pandas data frame containing anime recommendations
        filename: Name of wandb artifact to create
        translated: cleaned anime name
    """
    sypnopsis_df = get_sypnopses_df()
    model = get_model()
    weights, _ = get_weights(model)
    rating_df, anime_to_index, index_to_anime = main_df_by_anime()
    use_types = get_types()

    translated = clean(name)
    filename = translated + '.csv'
    logger.info('filename is %s', filename)

    # Get encoded index of input anime
    try:
        index = get_anime_frame(translated, anime_df).anime_id.values[0]
    except IndexError:
        try:
            # In case the name has special characters
            index = get_anime_frame(name, anime_df).anime_id.values[0]
        except IndexError:
            # In case there is a punctuation typo in the config file
            index = get_anime_frame(
                translated, anime_df, clean=True).anime_id.values[0]

    # Get input anime's index
    encoded_index = anime_to_index.get(index)
    # Take dot product of weights array and input anime's embedding vector to
    # Find cosine similarity of each. Higher values indicate closer similarity
    # weights.shape is (num anime, embedding length) e.g. (17560, 128)
    # weights[encoded_index].shape is (1, embedding length)
    dists = np.dot(weights, weights[encoded_index])

    # Get indices of values that are the highest dists and sort
    # E.g. value [20, 10, 40, 50, 30] -->  index [1, 0, 4, 2, 3]
    sorted_dists = np.argsort(dists)
    closest = sorted_dists[:]
    arr = []

    # Sequentially append closest animes to empty array
    for close in closest:
        # Get anime associated with dist index
        decoded_id = index_to_anime.get(close)
        # Get df of anime stats
        anime_frame = get_anime_frame(decoded_id, anime_df)

        # Some anime do not have sypnopses
        try:
            sypnopsis = get_sypnopsis(decoded_id, sypnopsis_df)
        except IndexError:
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
    if args.an_spec_genres is True:
        Frame = by_genre(Frame).sort_values(by="Similarity", ascending=False)
    else:
        Frame = Frame.sort_values(by="Similarity", ascending=False)

    try:
        return Frame[:count], filename, translated
    # If there aren't enough (count) similar anime, return all similar anime
    except IndexError:
        return Frame[:]


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
        metadata={"Queried anime": anime_name,
                  "Model used": args.model,
                  "Main data frame used": args.main_df,
                  "Filename": fn})

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
        "--anime_rec_genres",
        type=str,
        help="List of genres to narrow down return values",
        required=True
    )

    parser.add_argument(
        "--an_spec_genres",
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
