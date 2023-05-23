import re
import ast
import random
from distutils.util import strtobool
import numpy as np
import pandas as pd
import string
import wandb
import logging
import argparse
from helper_functions.load import main_df_by_anime, get_anime_df, get_model
from helper_functions.load import get_weights, get_sypnopses_df
from helper_functions.load import get_anime_frame, get_sypnopsis, get_genres
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))


logging.basicConfig(
    filename='./similar_anime.log',  # Path to log file
    level=logging.INFO,  # Log info, warnings, errors, and critical errors
    filemode='a',  # Create log file if one doesn't already exist and add
    format='%(asctime)s-%(name)s - %(levelname)s - %(message)s',
    datefmt='%d %b %Y %H:%M:%S %Z',  # Format date
    force=True)
logger = logging.getLogger()


def get_random_anime(anime_df):
    """
    Get a random anime from anime data frame
    """
    # anime_df = get_anime_df()
    possible_anime = anime_df['eng_version'].unique().tolist()
    random_anime = random.choice(possible_anime)
    return random_anime


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


def anime_recommendations(name, count, anime_df, sypnopsis_df, model, weights,
                          rating_df, anime_to_index, index_to_anime):
    """
    Get anime recommendations based on similar anime.
    Count is the number of similar anime to return based on highest score
    """
    use_types = get_types(anime_df)

    # Strip all Escape characters and spaces & produce filename
    translated = str(name).translate(
        {ord(c): None for c in string.whitespace})
    translated = re.sub(r'\W+', '', translated)
    filename = translated + '.csv'

    # Get ID and encoded index of input anime
    index = get_anime_frame(name, anime_df).anime_id.values[0]
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
        anime_name = anime_frame['eng_version'].values[0]
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
                    {"anime_id": decoded_id, "Name": anime_name,
                     "Similarity": similarity, "Genres": genre,
                     'Sypnopsis': sypnopsis, "Episodes": episodes,
                     "Japanese name": japanese_name, "Studios": studios,
                     "Premiered": premiered, "Score": score,
                     "Type": Type, "Source": source, 'Rating': rating})
        else:
            arr.append(
                {"anime_id": decoded_id, "Name": anime_name,
                 "Similarity": similarity, "Genres": genre,
                 'Sypnopsis': sypnopsis, "Episodes": episodes,
                 "Japanese name": japanese_name, "Studios": studios,
                 "Premiered": premiered, "Score": score,
                 "Type": Type, "Source": source, 'Rating': rating})

    # Convert array to data frame
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
    run = wandb.init(
        project=args.project_name,
        name="similar_anime")
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
    weights, _ = get_weights(model)
    rating_df, anime_to_index, index_to_anime = main_df_by_anime(
        project=args.project_name,
        main_df=args.main_df,
        artifact_type='preprocessed_data')

    if args.random_anime is True:
        anime_name = get_random_anime(anime_df)
        logger.info("Using %s as random input anime", anime_name)
    else:
        anime_name = args.anime_query

    # Create data frame file
    df, filename, name = anime_recommendations(
        anime_name,
        int(args.a_query_number),
        anime_df,
        sypnopsis_df,
        model,
        weights,
        rating_df,
        anime_to_index,
        index_to_anime)

    # Strip all Escape characters and spaces
    df.to_csv(filename, index=False)

    # Create artifact
    logger.info("Creating artifact")
    description = "Anime most similar to: " + str(anime_name)
    artifact = wandb.Artifact(
        name=filename,
        type="csv",
        description=description,
        metadata={"Queried anime: ": anime_name})

    # Upload artifact to wandb
    artifact.add_file(filename)
    logger.info("Logging artifact for anime %s", anime_name)
    run.log_artifact(artifact)
    artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get recommendations based on similar anime",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--weights",
        type=str,
        help="Wandb artifact with .h5 file of all neural network weights",
        required=True
    )

    parser.add_argument(
        "--history",
        type=str,
        help="Wandb artifact with .csv file of neural network run history",
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

    args = parser.parse_args()
    go(args)
