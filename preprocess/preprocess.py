import argparse
import logging
import os
import tempfile
import wandb
from distutils.util import strtobool
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def drop_useless(df):
    """
    String Input the filename of a data frame artifact
    Drop samples that are not appropriate to use.
    1) If no episodes had been watched
    2) If the user has rated less than a predetermined number of ratings
        Note: The default value of 50 was determined subjectively. Increasing
        it will shrink the amount of data at the cost of including users who
        aren't entirely active
    3) If no rating was assigned
    4) "Plan to watch" watching status
    """
    df = df.drop_duplicates()
    df = df.dropna()
    # Drop unwatched samples
    df = df[df['watched_episodes'] != 0]

    # Drop planned to watch
    df = df[df['watching_status'] != 6]

    # Drop users with an insufficient number of ratings
    # n_ratings is a Pandas Series showing the # reviews associated w/a userid
    n_ratings = df['user_id'].value_counts(dropna=True)
    df = df[df['user_id'].isin(
        n_ratings[n_ratings >= args.num_reviews].index)].copy()
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
    Since we are removing data, the model may score poorer; however, it will
    ultimately produce results that are more aligned with our goal.
    Due to the size of the data set, it is included as an optional argument in
    the MLflow config file.
    """
    # Create data frame containing total number of episodes each anime has
    temp_df = df.drop(['user_id', 'watching_status', 'rating'], axis=1)
    temp_df = temp_df.groupby('anime_id')['watched_episodes'].max()

    # Createa tuples of the form (anime_id, tot_episodes), then make hashable
    max_tuples = list(zip(temp_df.index, temp_df))
    max_dict = {}
    max_dict = Convert(max_tuples, max_dict)

    max_eps = []
    for sample in df.index:
        ID = df['anime_id'][sample]
        mx = max_dict.get(ID)[0]
        max_eps.append(mx)

    # Create a list of half the maximum episodes for each anime
    acceptable_tuples = []
    for (x, y) in max_tuples:
        # Movies and many OVAs only have one episode, so keep these
        if y == 1:
            acceptable_tuples.append((x, y))
        else:
            acceptable_tuples.append((x, y * .5))

    # Convert tuples list into hashable format
    dictionary = {}
    dictionary = Convert(acceptable_tuples, dictionary)
    # Create a list of half episodes for each sample in data frame
    half = []
    for sample in df.index:
        # ID is the Id of the anime
        ID = df['anime_id'][sample]
        # Half_eps is half the number of total episodes as found in dictionary
        half_eps = dictionary.get(ID)[0]
        half.append(half_eps)

    # Convert list into new data frame column
    df['max_eps'] = max_eps
    df['half_eps'] = half

    # Create data frame of only samples who have watched more than
    # or equal to half the number of available episodes
    df = df[df['watched_episodes'] >= df['half_eps']]
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


def go(args):

    # Instantiate wandb, run, and get raw data artifact
    run = wandb.init(
        job_type="preprocess_data",
        project=args.project_name)
    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.raw_stats, type='raw_data')
    artifact_path = artifact.file()
    df = pd.read_parquet(artifact_path, low_memory=False)

    # Drop unused features, duplicates, and outliers
    logger.info("Dropping useless data")
    df = drop_useless(df)

    if args.drop_half_watched is True:
        logger.info("Dropping samples with too few episodes watched")
        df = drop_half_watched(df)

    logger.info("Scaling ratings")
    df = scale_ratings(df)
    filename = args.preprocessed_stats
    curr_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp_dir:

        # Save clean df to local machine if desired
        if args.save_clean_locally is True:
            df2 = df.copy()
            local = 'preprocessed_data.parquet'
            df2.to_parquet(os.path.join(curr_dir, local))
        df.to_parquet(
            os.path.join(tmp_dir, args.preprocessed_stats), index=False)

        # Create artifact and upload to wandb
        artifact = wandb.Artifact(
            name=args.preprocessed_stats,
            type=args.preprocessed_artifact_type,
            description=args.preprocessed_artifact_description,
            metadata={"Was data saved locally?": args.save_clean_locally}
        )

        artifact.add_file(filename)
        logger.info("Logging artifact")
        run.log_artifact(artifact)
        artifact.wait()

    os.remove(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--raw_stats",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--project_name",
        type=str,
        help="Name of wandb project",
        required=True
    )

    parser.add_argument(
        "--preprocessed_stats",
        type=str,
        help="Name for the artifact",
        required=True
    )

    parser.add_argument(
        "--preprocessed_artifact_type",
        type=str,
        help="Type of preprocessed artifact",
        required=True
    )

    parser.add_argument(
        "--preprocessed_artifact_description",
        type=str,
        help="Description of preprocessed data artifact",
        required=True
    )

    parser.add_argument(
        "--num_reviews",
        type=str,
        help="minimum number of reviews a user should have",
        required=True
    )

    parser.add_argument(
        "--drop_half_watched",
        type=lambda x: bool(strtobool(x)),
        help="Drop sample if less than half the total episodes were watched",
        required=True
    )

    parser.add_argument(
        "--save_clean_locally",
        type=lambda x: bool(strtobool(x)),
        help='Choose whether or not to save clean data frame to local file',
        required=True
    )

    args = parser.parse_args()
    go(args)
