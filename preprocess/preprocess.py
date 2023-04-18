import argparse
import logging
# import os
from distutils.util import strtobool
# import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def drop_useless(df):
    """
    String Input the filename of a data frame artifact
    Drop samples that are not appropriate to use.
    1) If no episodes had been watched
    2) If the user has rated less than 350 anime
    3) If no rating was assigned
    4) "Plan to watch" watching status
    """
    # local_df = pd.read_parquet(os.path.join(os.getcwd(), df))
    df = df.drop_duplicates()
    df = df.dropna()
    # Drop unwatched samples
    df = df[df['watched_episodes'] != 0]

    # Drop planned to watch
    df = df[df['watching_status'] != 6]

    # Drop users with an insufficient number of ratings
    # n_ratings is a Pandas Series showing the # reviews associated w/a userid
    n_ratings = df['user_id'].value_counts(dropna=True)
    df = df[df['user_id'].isin(n_ratings[n_ratings >= 350].index)].copy()
    return df


def Convert(tup, di):
    """
    Helper function for converting tuples to dictionaries
    """
    for a, b in tup:
        di.setdefault(a, []).append(b)
    return di


def drop_half_watched(df):
    """
    We have already dropped rows with 0 watched episodes. This expands on that
    by dropping samples where a user has viewed under half the total episodes.
    This ensures the reviewer has an appropriate understanding of the show.
    It takes a while due to the size of the dataset, so it is included as an
    optional argument in the MLflow config file.
    Since we are removing data, the model may score poorer; however, it will
    ultimately produce results that are more aligned with our goal.
    """
    # First get total episodes of each anime
    temp_df = df.drop(['user_id', 'watching_status', 'rating'], axis=1)
    temp_df = temp_df.groupby('anime_id')['watched_episodes'].max()
    max_tuples = list(zip(temp_df.index, temp_df))  # (anime_id, tot_episodes)

    # Make a list of half the max episodes for each anime
    acceptable_tuples = []
    for (x, y) in max_tuples:
        # Movies and many OVAs only have one episode, so exclude these
        if y == 1:
            acceptable_tuples.append((x, y))
        else:
            acceptable_tuples.append((x, y * .5))

    # Convert tuples list into hashable format
    dictionary = {}
    dictionary = Convert(acceptable_tuples, dictionary)

    # For each sample in the data frame
    for sample in df.index:
        # ID is the Id of the anime
        ID = df['anime_id'][sample]
        # Half_eps is half the number of total episodes
        half_eps = dictionary.get(ID)[0]
        # Watched is the number of episodes watched
        watched = float(df['watched_episodes'][sample])
        # Remove row if less than half the total episodes were watched
        if watched <= half_eps and half_eps >= 1.0:
            df = df.drop(sample, axis=0)
    return df


def scale_ratings(df):
    """
    Scale ratings to between 0 and 1
    """
    min_rating = min(df['rating'])
    max_rating = max(df['rating'])

    df['rating'] = df['rating'].apply(lambda x: (
        x - min_rating) / (max_rating - min_rating)).values.astype(np.float64)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--output_name",
        type=str,
        help="Name for the artifact",
        required=True
    )

    parser.add_argument(
        "--drop_half_watched",
        type=lambda x: bool(strtobool(x)),
        help="Drop sample if less than half the total episodes were watched",
        required=True
    )

    # True values are y, yes, t, true, on and 1;
    # False values are n, no, f, false, off and 0
    # Will raise ValueError if input argument is not of proper type
    parser.add_argument(
        "--save_clean_locally",
        type=lambda x: bool(strtobool(x)),
        help='Choose whether or not to save clean data frame to local file',
        required=True
    )

    args = parser.parse_args()

#    go(args)
