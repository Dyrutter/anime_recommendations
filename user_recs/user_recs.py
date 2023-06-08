import unicodedata
import string
import argparse
import logging
import os
import wandb
import pandas as pd
from distutils.util import strtobool
import re
import numpy as np
import tensorflow as tf
import random
from wordcloud import WordCloud


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


def main_df_by_id():
    """
    Load main data frame artifact from wandb
    Outputs:
        df: Main Pandas Data Frame of user stats, keeping the columns
            "user_id", "anime_id", and "rating", as well as adding mapped
            columns "user" and "anime"
        user_to_index: enumerated IDs of format {ID: enumerated_index}
        index_to_user: enumerated IDs of format {enumerated_index: ID}
    """
    run = wandb.init(project=args.project_name)
    artifact = run.use_artifact(args.main_df, type=args.main_df_type)
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


def get_fave_df(genres, sources, ID):
    """
    Merge favorite sources and favorite genres data frames
    Inputs:
        genres: Pandas data frame of a user's favorite anime and their 
            respective genres with columns ["eng_version", "Genres"]
        sources: Pandas data frame of a user's favorite anime and their
            respective source material with columns ["eng_version", "sources"]
    Outputs:
        sources: Merged data frame with columns
            ["eng_version", "Genres", "Sources"]
        fn: Filename to save Data Frame under
    """
    genres = genres["Genres"]
    sources["Genres"] = genres
    fn = 'User_ID_' + str(ID) + '_user_recs_faves.csv'
    sources.to_csv(fn)
    return sources, fn


def get_anime_frame(anime, df, clean=False):
    """
    Helper function to get a specific anime in data frame format
    Input:
        anime: Either the string name of the anime or an int of the anime's ID
        df: Data Frame containing list of all anime
        clean: If True, return the name of the anime cleaned with clean()
           If False, return the anime's full name
    Output:
        df: An anime's name or ID in data frame format
    """
    if isinstance(anime, int):
        return df[df.anime_id == anime]
    if isinstance(anime, str):
        if clean is False:
            return df[df.Name == anime]
        else:
            return df[df.eng_version == anime]


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
    # Get list of possible user IDs
    possible_users = list(user_to_index.keys())
    # Select random user from list of IDs 
    random_user = random.choice(possible_users)
    return random_user


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


def get_sources(anime_df):
    """
    Get all possible anime sources.
    Input: data frame containing anime statistics taken from get_anime_df()
    Output: All possible anime sources in list format
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


def fave_genres(user, df, anime_df):
    """
    Get the favorite genres of a user
    Inputs:
        user: an integer user ID
        df: main df of cols ['user', 'anime', 'rating', 'user_id', 'anime_id']
        anime_df: df containing anime statisitics
    Outputs:
        faves: Pandas data frame containing a user's favorite anime and their
           respective genres
    """
    watched = df[df.user_id == user]
    user_rating_percentile = np.percentile(watched.rating, 80)
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
        faves: Pandas data frame containing a user's favorite anime and their
           respective sources
    """
    watched = df[df.user_id == user]
    user_rating_percentile = np.percentile(watched.rating, 80)
    watched = watched[watched.rating >= user_rating_percentile]
    top = (watched.sort_values(by="rating", ascending=False).anime_id.values)

    faves = anime_df[anime_df["anime_id"].isin(top)]
    faves = faves[["eng_version", "Source"]]
    return pd.DataFrame(faves)


def genre_cloud(anime_df, ID):
    """
    Create a word cloud of a user's favorite genres
    Inputs:
        anime_df: anime stats data frame
        ID: User ID to create cloud of
    Outputs:
        genres_cloud: A wordcloud object of the user's favorite genres
        fn: Filename wordcloud is saved under
    """
    genres = get_genres(anime_df)
    genres = (" ").join(list(map(str.upper, genres)))

    cloud = WordCloud(width=800,
                      height=600,
                      prefer_horizontal=0.85,
                      background_color='white',
                      contour_width=0.05,
                      colormap='spring')
    genres_cloud = cloud.generate(genres)
    fn = "User_ID_" + str(ID) + '_user_recs_genre_cloud.png'
    genres_cloud.to_file(fn)
    return genres_cloud, fn


def source_cloud(anime_df, ID):
    """
    Create a word cloud of a user's favorite sources
    Inputs:
        anime_df: anime stats data frame
        ID: User ID to create cloud of
    Outputs:
        source_cloud: a wordcloud object of the user's favorite sources
        fn: The filename the word cloud is saved under
    """
    source = get_sources(anime_df)
    sources = (" ").join(list(map(str.upper, source)))

    cloud = WordCloud(width=800,
                      height=600,
                      prefer_horizontal=0.85,
                      background_color='white',
                      contour_width=0.05,
                      colormap='spring')
    source_cloud = cloud.generate(sources)
    fn = 'User_ID_' + str(ID) + '_user_recs_source_cloud.png'
    source_cloud.to_file(fn)
    return source_cloud, fn


def find_similar_users(user_id, n_users, rating_df, user_to_index,
                       index_to_user, user_weights):
    """
    Find similar IDs to an input IDs. This function is called if
    args.recs_sim_from_flow is False, meaning no similar users Data Frame
    artifact was imported from wandb.
    Inputs:
        user_id: Int, the ID of which to find similar users to.
            If args.recs_ID_from_conf is True, input args.user_recs_query ID
            If args.user_recs_random is True, input a random ID
        n_users: Int, the number of similar users to find
        rating_df: Main Pandas rating data frame
        user_to_index: dict, enumerated mapping taken from main_df_by_id()
        index_to_user: dict, enumerated mapping taken from main_df_by_id()
        user_weights: np array of user weights array taken from get_weights()
    Outputs:
        Frame: Pandas data frame of similar users with columns "similar_users"
            and "similarity"
    """
    index = user_id
    encoded_index = user_to_index.get(index)

    dists = np.dot(user_weights, user_weights[encoded_index])
    sorted_dists = np.argsort(dists)
    n_users = n_users + 1
    closest = sorted_dists[-n_users:]

    sm_arr = []

    for close in closest:
        similarity = dists[close]
        decoded_id = index_to_user.get(close)
        if decoded_id != user_id:
            sm_arr.append(
                {"similar_users": decoded_id,
                 "similarity": similarity})

    Frame = pd.DataFrame(sm_arr).sort_values(by="similarity", ascending=False)

    return Frame


def get_ID_artifacts():
    """
    Get the user ID and similar users artifact created in similar_users.py
    Outputs:
        ID: Integer user ID taken from MLflow
        sim_IDs: data frame of IDs similar to ID
    """
    run = wandb.init(project=args.project_name)
    artifact = run.use_artifact(args.flow_ID, type=args.flow_ID_type)
    artifact_path = artifact.file()
    ID_df = pd.read_csv(artifact_path)
    ID = ID_df.values[0][0]

    sim_artifact = run.use_artifact(
        args.sim_users_art, type=args.sim_users_art_type)
    sim_art_path = sim_artifact.file()
    sim_IDs = pd.read_csv(sim_art_path)
    return ID, sim_IDs


def select_user(num_sim, df, user_to_index, index_to_user, user_weights):
    """
    Choose user to analyze based. Will be either the ID used
        in the MLflow workflow, or the ID input in the config file under
        the section ["Users"]["user_recs_query"], or a random ID
    Inputs:
        num_sim: int, num of similar users to return
        df: Main df taken from main_df_by_id()
        user_to_index: dict, enumerated mapping taken from main_df_by_id()
        index_to_user: dict, enumerated mapping taken from main_df_by_id()
        user_weights: np array of user weights array taken from get_weights()
    Outputs:
        User_ID: Integer of the user ID to analzye
        sim_df: Pandas Data Frame of Similar IDs with columns "similar_users"
             and "similarity"
    """
    if args.recs_ID_from_flow is True:
        user, sim_df = get_ID_artifacts()
        logger.info("Using %s as input use taken from MLflow", user)
        return user, sim_df

    elif args.recs_ID_from_conf is True:
        user = int(args.user_recs_query)
        sim_df = find_similar_users(
            user, num_sim, df, user_to_index, index_to_user, user_weights)
        logger.info("Using %s as config file-specified input user", user)
        return user, sim_df
    else:
        user = get_random_user()
        logger.info("Using %s as random input user", user)
        sim_df = find_similar_users(
            user, num_sim, df, user_to_index, index_to_user, user_weights)
        return user, sim_df


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
    """
    Recommend anime to a user based on anime that similar IDs have favoirted.
    The more commonly favorited those anime were by similar users, the higher
    they are ranked in the recommendation. For example, if 9 out of 10 similar
    users had favorited Black Lagoon, Black Lagoon would be highly ranked.
    Inputs:
        user: Int, the user ID to find recommendations for
        similar_users: Pandas Data Frame that must include the columns
            "similar_users" and "similarity"
        sypnopsis_df: The sypnopsis data frame taken from get_sypnopsis_df()
        rating_df: The rating data frame taken from main_df_by_ID()
        user_to_index: Dictionary, mapping of user IDs to indexes in rating_df
        index_to_user: Dictionary, mapping of indexes to user IDs in rating_df
        anime_df: Pandas Data Frame of all anime taken from get_anime_df()
        n: Int, number of anime recommendations to return
        genre_df: Pandas Data Frame of favorite genres from fave_genres()
        source_df: Pandas Data Frame of favorite sources from fave_sources()
    Outputs:
        df: Data Frame of anime recommendations ranked by similar users
        filename: Name of wandb artifact to save df as
    """
    user_pref, _ = get_fave_df(genre_df, source_df, user)
    recommended_animes, anime_list = [], []
    eng_versions = user_pref.eng_version.values.tolist()

    # Get list of similar user IDs
    for user_id in similar_users.similar_users.values:
        genre_df = fave_genres(user_id, rating_df, anime_df)
        source_df = fave_sources(user_id, rating_df, anime_df)
        pref_list, _ = get_fave_df(genre_df, source_df, user)
        # Favorites of similar users that input user has not watched
        pref_list = pref_list[~pref_list.eng_version.isin(eng_versions)]
        anime_list.append(pref_list.eng_version.values)

    anime_list = pd.DataFrame(anime_list)
    sorted_list = pd.DataFrame(
        pd.Series(anime_list.values.ravel()).value_counts()).head(n)
    logger.info("anime name is %s", sorted_list)

    for i, anime_name in enumerate(sorted_list.index):
        n_pref = sorted_list[sorted_list.index == anime_name].values[0][0]
        anime_frame = get_anime_frame(anime_name, anime_df, clean=True)
        name = anime_frame['Name'].values[0]
        genre = anime_frame['Genres'].values[0]

        japanese_name = anime_frame['japanese_name'].values[0]
        episodes = anime_frame['Episodes'].values[0]
        premiered = anime_frame['Premiered'].values[0]
        studios = anime_frame['Studios'].values[0]
        score = anime_frame["Score"].values[0]
        Type = anime_frame['Type'].values[0]
        source = anime_frame['Source'].values[0]
        anime_id = anime_frame['anime_id'].values[0]

        # Some anime do not have a sypnopsis included
        try:
            sypnopsis = get_sypnopsis(int(anime_id), sypnopsis_df)
        except IndexError:
            sypnopsis = "None"
        recommended_animes.append(
            {"anime_id": anime_id, "Name": name, "n_user_prefs": n_pref,
             "Source": source, "Genres": genre,
             'Sypnopsis': sypnopsis, "Episodes": episodes,
             "Japanese name": japanese_name, "Studios": studios,
             "Premiered": premiered, "Score": score, "Type": Type})
    filename = 'User_ID_' + str(user) + '_' + args.user_recs_fn
    df = pd.DataFrame(recommended_animes)
    return df, filename


def go(args):
    # Initialize run
    run = wandb.init(project=args.project_name,
                     name="user_genre_based_preferences_recommendations")
    df, ID_to_index, index_to_user = main_df_by_id()
    anime_df = get_anime_df()
    sypnopsis_df = get_sypnopses_df()
    model = get_model()
    anime_weights, user_weights = get_weights(model)
    user, similar_users = select_user(
        int(args.recs_n_sim_ID), df, ID_to_index, index_to_user, user_weights)
    genre_df = fave_genres(user, df, anime_df)
    source_df = fave_sources(user, df, anime_df)

    users_recs_df, filename = similar_user_recs(
        user, similar_users, sypnopsis_df, df, ID_to_index,
        index_to_user, anime_df, int(args.user_num_recs), genre_df, source_df)
    users_recs_df.to_csv(filename, index=False)

    genres_cloud, genre_fn = genre_cloud(anime_df, user)
    sources_cloud, source_fn = source_cloud(anime_df, user)
    fave_df, fave_fn = get_fave_df(genre_df, source_df, user)

    # Create artifact
    logger.info("Creating similar user based recs artifact")
    description = "Anime recs based on user prefs: " + str(user)
    artifact = wandb.Artifact(
        name=filename,
        type=args.user_recs_type,
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
        "--user_recs_query",
        type=str,
        help="input user id to query",
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
        "--model",
        type=str,
        help="Wandb artifact with .h5 file of neural network",
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
        "--user_recs_random",
        type=lambda x: bool(strtobool(x)),
        help="Boolean, whether to user a random user",
        required=True
    )

    parser.add_argument(
        "--recs_sim_from_flow",
        type=lambda x: bool(strtobool(x)),
        help="Boolean, whether to use similar users artifact",
        required=True
    )

    parser.add_argument(
        "--user_recs_type",
        type=str,
        help="Type of user recs artifact to create",
        required=True
    )

    parser.add_argument(
        "--recs_ID_from_flow",
        type=lambda x: bool(strtobool(x)),
        help="Whether to use the User ID artifact created in MLflow",
        required=True
    )

    parser.add_argument(
        "--flow_ID",
        type=str,
        help='ID of MLflow artifact name to use if recs_ID_from_flow is True',
        required=True
    )

    parser.add_argument(
        "--flow_ID_type",
        type=str,
        help="Type of mlflow artifact user ID was saved as",
        required=True
    )

    parser.add_argument(
        "--sim_users_art",
        type=str,
        help="Name of similar users artifact if recs_sim_from_flow is True",
        required=True
    )

    parser.add_argument(
        "--sim_users_art_type",
        type=str,
        help="Type of similar users artifact to load",
        required=True
    )

    parser.add_argument(
        "--recs_n_sim_ID",
        type=str,
        help="Number of similar users to include",
        required=True
    )

    parser.add_argument(
        "--recs_ID_from_conf",
        type=lambda x: bool(strtobool(x)),
        help="Whether to use the user ID from config file user_recs_query",
        required=True
    )
    args = parser.parse_args()
    go(args)
