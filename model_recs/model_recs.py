import argparse
import logging
import os
import wandb
import string
import pandas as pd
from distutils.util import strtobool
import re
import random
import numpy as np
import tensorflow as tf
import ast
import unicodedata

logging.basicConfig(
    filename='./model_recs.log',
    level=logging.INFO,
    filemode='a',
    format='%(asctime)s-%(name)s - %(levelname)s - %(message)s',
    datefmt='%d %b %Y %H:%M:%S %Z',
    force=True)
logger = logging.getLogger()


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


def get_full_df():
    """
    Load main data frame artifact from wandb
    Outputs:
        df: Main Pandas Data Frame of user stats, keeping the columns
            "user_id", "anime_id", and "rating", as well as adding mapped
            columns "user" and "anime"
    """
    run = wandb.init(project=args.project_name)
    # logger.info("Downloading data artifact")
    artifact = run.use_artifact(args.main_df, type=args.main_df_type)
    artifact_path = artifact.file()
    df = pd.read_parquet(artifact_path)

    # Encoding categorical data, get list of all anime ids and user ids
    user_ids = df["user_id"].unique().tolist()
    anime_ids = df["anime_id"].unique().tolist()

    # Dicts of format {id: count_number}
    anime_to_index = {value: count for count, value in enumerate(anime_ids)}
    user_to_index = {value: count for count, value in enumerate(user_ids)}

    # Convert values of format id to count_number
    df["user"] = df["user_id"].map(user_to_index)
    df["anime"] = df["anime_id"].map(anime_to_index)
    df = df[['user', 'anime', 'rating', 'anime_id', 'user_id']]
    df = df.sample(frac=1, random_state=42)
    return df


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

    # Add "anime_id" column and remove spaces from column names
    df['anime_id'] = df['MAL_ID']
    df['japanese_name'] = df['Japanese name']
    df["eng_version"] = df['English name']

    # Get column of cleaned anime names
    df['eng_version'] = df.anime_id.apply(
        lambda x: clean(get_anime_name(x, df)).lower())
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


def get_unwatched(df, anime_df, user):
    """
    Get array of anime a user hasn't watched
    Inputs:
        df: Main Pandas Data Frame of user stats with columns
            ["user", "anime", "rating", "anime_id", "user_id"]
        anime_df: Pandas Data Frame with all anime and their relevant stats
        user: Int, The User ID to query
    Outputs:
        unwatched: array containing the anime IDs of all unwatched anime
    """
    # Get df of user rating instances, shape [number of ratings, df columns]
    watched = df[df.user_id == user]
    # Reduce anime data frame to include only anime the user hasn't watched
    unwatched = anime_df[~anime_df['anime_id'].isin(watched.anime_id.values)]
    # Get list of anime IDs the user hasn't watched
    anime_ids = df["anime_id"].unique().tolist()
    # Create dict mapping {anime_id: index}
    anime_to_index = {value: count for count, value in enumerate(anime_ids)}
    # Get list of IDs that are unwatched intersecting with existing anime IDs
    unwatched = list(set(unwatched['anime_id']).intersection(
        set(anime_to_index.keys())))
    # Get array of indices for unwatched animes from anime_to_index dict
    unwatched = [[anime_to_index.get(x)] for x in unwatched]
    return unwatched


def get_user_anime_arr(df, anime_df, user, unwatched):
    """
    Get array of anime a user hasn't watched formatted for neural network
        predictions.
    Inputs:
        df: Main Pandas Data Frame of user stats with columns
            ["user", "anime", "rating", "anime_id", "user_id"]
        anime_df: Pandas Data Frame with all anime and their relevant stats
        user: Int, The User ID to query
        unwatched: Nested list of indices of enumerated unwatched anime IDs
            [[anime index], [anime_index], [anime_index]...]
    Outputs:
        unwatched: list of two 1D arrays, the first a repeated list of the
            input user's index among an enumerated list of all unique user
            IDs found in df, the second a list of the enumerated
            indices of animes the user hasn't watched, formatted as:
            [array([user_ind, user_ind...]), array([anime_ind, anime_ind]...)]
    """
    # Get list of unique user IDs
    user_ids = df["user_id"].unique().tolist()
    # Get dict mapping of {user ID: Index}
    user_to_index = {value: count for count, value in enumerate(user_ids)}
    # Get user's associated index
    user_encoded = user_to_index.get(user)
    # Get array of the repeated user ID with length of # of unwatched anime
    same_user_arr = [[user_encoded]] * len(unwatched)
    # Get 2D array of format [[user_ind, anime_ind], [user_ind, anime_ind]...]
    combined = np.hstack((same_user_arr, unwatched))
    # Create 1D array of format array([user_ind, user_ind, user_ind...])
    user_arr = combined[:, 0]
    # Create 1D array of format array([anime_ind, anime_ind, anime_ind...])
    unwatched_arr = combined[:, 1]
    # Return list containing user array and unwatched anime array
    return [user_arr, unwatched_arr]


def get_model():
    """
    Download neural network h5 model artifact from wandb
    Output:
        model: TensorFlow neural network model
    """
    run = wandb.init(project=args.project_name)
    artifact = run.use_artifact(args.model, type=args.model_type)
    artifact_path = artifact.file()
    model = tf.keras.models.load_model(artifact_path)
    return model


def get_sypnopses_df():
    """
    Download sypnopses data frame artifact from wandb
    Output:
        df: Pandas Data Frame containting columns
            ["MAL_ID", "Name", "Genres", "sypnopsis"]
    """
    run = wandb.init(project=args.project_name)
    art = run.use_artifact(args.sypnopsis_df, type=args.sypnopsis_df_type)
    artifact_path = art.file()
    cols = ["MAL_ID", "Name", "Genres", "sypnopsis"]
    df = pd.read_csv(artifact_path, usecols=cols)
    return df


def get_random_user(df):
    """
    Get a random user from main data frame
    Inputs:
        df: main df of cols ['user', 'anime', 'rating', 'user_id', 'anime_id']
    Outputs:
        random_user: Integer value of a random user's ID
    """
    # Get unique user IDs
    user_ids = df["user_id"].unique().tolist()
    # Get dict mapping User IDs to their Index {ID: Index}
    user_to_index = {value: count for count, value in enumerate(user_ids)}
    # Get list of keys (IDs) and select a random ID
    possible_users = list(user_to_index.keys())
    random_user = int(random.choice(possible_users))
    return random_user


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


def get_genres(anime_df):
    """
    Get all possible anime genres
    Input: data frame containing anime statistics taken from get_anime_df()
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


def by_genre(anime_df):
    """
    Restrict the potential anime recommendations according to genre
    Inputs:
        anime_df: Pandas Data Frame with all anime and their relevant stats
    Output:
        df: Anime data frame modified to include only anime of the genre(s)
           specified under args.model_genres
    """
    # Get genres to use and possible genres
    use_genres = clean(ast.literal_eval(args.model_genres))
    genres = clean(get_genres(anime_df))
    # Ensure the input genres are valid genres
    for genre in use_genres:
        try:
            assert genre in clean(genres)
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


def get_ID_artifact():
    """
    Get the user ID and similar users artifact created in similar_users.py
    Outputs:
        ID: Integer user ID taken from MLflow
    """
    run = wandb.init(project=args.project_name)
    artifact = run.use_artifact(args.flow_ID, type=args.flow_ID_type)
    artifact_path = artifact.file()
    ID_df = pd.read_csv(artifact_path)
    ID = ID_df.values[0][0]
    return ID


def select_user(df):
    """
    Choose user to analyze. Will be either the ID used in the MLflow
        workflow, the ID input in the config file under the section
        ["model_recs"]["model_user_query"], or a random ID
    Inputs:
        df: Main df taken from get_full_df()
    Outputs:
        User: Integer of the user ID to analzye
    """
    if args.model_ID_flow is True:
        user = get_ID_artifact()
        logger.info("Using %s as input use taken from MLflow", user)
        return user

    elif args.model_ID_conf is True:
        user = int(args.model_user_query)
        logger.info("Using %s as config file-specified input user", user)
        return user
    else:
        user = get_random_user()
        logger.info("Using %s as random input user", user)
        return user


def recommendations(df, anime_df, syp_df, model, id_anime, unwatched, n_recs):
    """
    Get anime recommendations based on model rating predictions. The higher
    the predcted rating (dot product of the estimated user and anime ID latent
    vectors established through gradient descent), the higher the anime is
    recommended. Only unwatched anime are included.
    Inputs:
        df: Main data frame taken from get_full_df()
        anime_df: Anime stats data frame taken from get_anime_df()
        syp_df: Sypnopses data frame taken from get_sypnopses_df()
        model: Neural network model taken from get_model()
        id_anime: ID/unwatched anime arrays taken from get_user_anime_arr()
        unwatched: unwatched anime array taken from get_unwatched()
        n_recs: Int, number of anime recommendations to return
    Outputs:
        Pandas data frame of anime recommendations
    """
    # Get unique anime IDs
    anime_ids = df["anime_id"].unique().tolist()
    # Input list of two arrays, first user ID indices, second anime ID indices
    ratings = model.predict(id_anime, verbose=2).flatten()
    # Get list of prediction indices sorrted based on top ratings
    top_inds = (-ratings).argsort()[:]
    # Index anime IDs for dictionary use
    index_to_anime = {index: anime for index, anime in enumerate(anime_ids)}
    # Get anime IDs that are unwatched
    rec_ids = [index_to_anime.get(unwatched[x][0]) for x in top_inds]
    anime_recs, top_ids = [], []
    # Get predicted ratings for unwatched anime using the anime's index
    for index, anime_id in enumerate(unwatched):
        # Ratings prediction value at a specific anime's index
        rating = ratings[index]
        # Anime ID associated with that index
        ID = index_to_anime.get(anime_id[0])
        # Append anime stats if the anime is among recommended anime
        if ID in rec_ids:
            top_ids.append(ID)
            try:
                condition = (anime_df.anime_id == ID)
                name = anime_df[condition]['Name'].values[0]
                genre = anime_df[condition]['Genres'].values[0]
                japanese_name = anime_df[condition]['japanese_name'].values[0]
                episodes = anime_df[condition]['Episodes'].values[0]
                premiered = anime_df[condition]['Premiered'].values[0]
                studios = anime_df[condition]["Studios"].values[0]
                score = anime_df[condition]["Score"].values[0]
                Type = anime_df[condition]["Type"].values[0]
                source = anime_df[condition]["Source"].values[0]
                anime_id = anime_df[condition]["anime_id"].values[0]
                # Some anime don't have sypnopses
                try:
                    sypnopsis = get_sypnopsis(int(anime_id), syp_df)
                except IndexError:
                    sypnopsis = "None"
            except IndexError:
                continue
            # If config file indicates to only return anime of specific types
            if args.specify_types is True:
                if Type in ast.literal_eval(args.anime_types):
                    anime_recs.append(
                        {"Name": name, "Prediction": rating, "Genres": genre,
                         "Source": source, "anime_id": anime_id,
                         'Sypnopsis': sypnopsis, "Episodes": episodes,
                         "Japanese name": japanese_name, "Studios": studios,
                         "Premiered": premiered, "Score": score, "Type": Type})
            else:
                anime_recs.append(
                    {"Name": name, "Prediciton_rating": rating,
                     "Source": source, "Genres": genre, "anime_id": anime_id,
                     'Sypnopsis': sypnopsis, "Episodes": episodes,
                     "Japanese name": japanese_name, "Studios": studios,
                     "Premiered": premiered, "Score": score, "Type": Type})
    Frame = pd.DataFrame(anime_recs)
    # Remove anime not of specified genre if config file asserts so
    if args.specify_genres is True:
        Frame = by_genre(Frame)
    # Sort anime data frame based on rating prediction
    Frame = Frame.sort_values(by="Prediction", ascending=False)
    # Return data frame containing n_recs recommendations
    try:
        return Frame[:n_recs]
    except IndexError:
        return Frame


def go(args):
    # Initialize run
    run = wandb.init(project=args.project_name)
    df = get_full_df()
    anime_df = get_anime_df()
    sypnopsis_df = get_sypnopses_df()
    model = get_model()
    user = select_user(df)

    unwatched = get_unwatched(df, anime_df, user)
    arr = get_user_anime_arr(df, anime_df, user, unwatched)
    recs = recommendations(
        df,
        anime_df,
        sypnopsis_df,
        model,
        arr,
        unwatched,
        int(args.model_num_recs))
    filename = 'User_ID_' + str(user) + '_' + args.model_recs_fn
    recs.to_csv(filename, index=False)

    # Create artifact
    logger.info("Creating artifact")
    description = "Anime recs based on model rankings for user : " + str(user)
    artifact = wandb.Artifact(
        name=args.model_recs_fn,
        type=args.model_recs_type,
        description=description,
        metadata={"Queried user: ": user, "Filename": filename})

    # Upload artifact to wandb
    artifact.add_file(filename)
    logger.info("Logging artifact for user %s", user)
    run.log_artifact(artifact)
    artifact.wait()

    if args.save_model_recs is False:
        os.remove(filename)


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
        "--main_df_type",
        type=str,
        help="Artifact type of main data frame",
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
        "--anime_df_type",
        type=str,
        help='Artifact type of anime df',
        required=True
    )

    parser.add_argument(
        "--sypnopsis_df",
        type=str,
        help="Sypnopses df",
        required=True
    )

    parser.add_argument(
        "--sypnopsis_df_type",
        type=str,
        help="Sypnopsis artifact type",
        required=True
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model artifact",
        required=True
    )

    parser.add_argument(
        "--model_type",
        type=str,
        help="Type of neural network artifact, default h5",
        required=True
    )

    parser.add_argument(
        "--model_user_query",
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
        "--model_recs_fn",
        type=str,
        help="Artifact name of model recommendations csv file",
        required=True
    )

    parser.add_argument(
        "--save_model_recs",
        type=lambda x: bool(strtobool(x)),
        help="Whether or not to save model recs file locally",
        required=True
    )

    parser.add_argument(
        "--model_num_recs",
        type=str,
        help="Number of anime recommendations to return",
        required=True
    )

    parser.add_argument(
        "--anime_types",
        type=str,
        help="List of anime types to use in recommendations ['TV', 'OVA'...]",
        required=True
    )

    parser.add_argument(
        "--specify_types",
        type=lambda x: bool(strtobool(x)),
        help="Boolean of whether or not to specify types of anime to return",
        required=True
    )

    parser.add_argument(
        "--model_genres",
        type=str,
        help="List of genres to include in return list",
        required=True
    )

    parser.add_argument(
        "--specify_genres",
        type=lambda x: bool(strtobool(x)),
        help="Boolean of whether or not to specify types of anime to return",
        required=True
    )

    parser.add_argument(
        "--model_ID_flow",
        type=lambda x: bool(strtobool(x)),
        help="Boolean of whether to use wandb user ID artifact",
        required=True
    )

    parser.add_argument(
        "--model_ID_conf",
        type=lambda x: bool(strtobool(x)),
        help="Boolean to use ID specified in config file model_user_query",
        required=True
    )

    parser.add_argument(
        "--model_recs_type",
        type=str,
        help="Type of wandb artifact to save model recommendations as",
        required=True
    )

    parser.add_argument(
        "--flow_ID",
        type=str,
        help='ID of MLflow artifact name to use if model_ID_flow is True',
        required=True
    )

    parser.add_argument(
        "--flow_ID_type",
        type=str,
        help="Type of mlflow artifact user ID was saved as",
        required=True
    )
    args = parser.parse_args()
    go(args)
