import argparse
import logging
import pathlib
import wandb
import requests
import tempfile
import os
import pandas as pd
from distutils.util import strtobool

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    # Derive the base name of the data file from the URL
    stats_basename = pathlib.Path(
        args.stats_url).name.split("?")[0].split("#")[0]
    synopses_basename = pathlib.Path(
        args.synopses_url).name.split("?")[0].split("#")[0]
    all_anime_basename = pathlib.Path(
        args.all_anime_url).name.split("?")[0].split("#")[0]

    logger.info(
        f"Downloading {args.stats}, {args.synopses}, and {args.all_anime}")

    stats_file = os.path.join(os.getcwd(), f"{args.stats_artifact}")
    synopses_file = os.path.join(os.getcwd(), f"{args.synopses_artifact}")
    all_anime_file = os.path.join(os.getcwd(), f"{args.all_anime_artifact}")

    # Create stats artifact
    # Download file, streaming so we can download files larger than
    # the available memory. Named temporary file gets destroyed at the end so
    # nothing is left behind and the file gets removed even in case of errors
    with tempfile.NamedTemporaryFile(mode='wb+') as fp:
        logger.info("Creating run for downloading stats file")
        with wandb.init(job_type="download_data") as run:
            # Download the file streaming and write to open temp file
            with requests.get(args.stats_url, stream=True) as r:
                for chunk in r.iter_content(chunk_size=8192):
                    fp.write(chunk)
            fp.flush()  # ensure file written to disk before uploading to w&B
            logger.info("Creating stats artifact")
            artifact = wandb.Artifact(
                name=args.stats_artifact,
                type=args.stats_artifact_type,
                description=args.stats_artifact_description,
                metadata={'stats_url': args.stats_url})
            artifact.add_file(fp.name, name=stats_basename)
            logger.info("Logging stats artifact")
            run.log_artifact(artifact)
            artifact.wait()

    # Create synopses artifact
    with tempfile.NamedTemporaryFile(mode='wb+') as fp:
        logger.info("Creating run for downloading synopses data")
        with wandb.init(job_type="download_data") as run:
            with requests.get(args.synopses_url, stream=True) as r:
                for chunk in r.iter_content(chunk_size=8192):
                    fp.write(chunk)
            fp.flush()
            logger.info("Creating synopses artifact")
            artifact = wandb.Artifact(
                name=args.synopses_artifact,
                type=args.synopses_artifact_type,
                description=args.synopses_artifact_description,
                metadata={'synopses_url': args.synopses_url})
            artifact.add_file(fp.name, name=synopses_basename)
            logger.info("Logging synopses artifact")
            run.log_artifact(artifact)
            artifact.wait()

    # Create all_anime artifact
    with tempfile.NamedTemporaryFile(mode='wb+') as fp:
        logger.info("Creating run for downloading all_anime data")
        with wandb.init(job_type="download_data") as run:
            with requests.get(args.all_anime_url, stream=True) as r:
                for chunk in r.iter_content(chunk_size=8192):
                    fp.write(chunk)
            fp.flush()
            logger.info("Creating all_anime artifact")
            artifact = wandb.Artifact(
                name=args.all_anime_artifact,
                type=args.all_anime_type,
                description=args.all_anime_description,
                metadata={'all_anime_url': args.all_anime_url})
            artifact.add_file(fp.name, name=all_anime_basename)
            logger.info("Logging all_anime artifact")
            run.log_artifact(artifact)
            artifact.wait()

    # Save files to local system if desired
    if args.save_raw_locally is True:
        with wandb.init(job_type='download_data') as run:
            stats_artifact = run.use_artifact(args.stats_artifact)
            stats_artifact_path = stats_artifact.file()
            stats_df = pd.read_parquet(stats_artifact_path, low_memory=False)
            stats_df.to_parquet(stats_file)

            synopses_artifact = run.use_artifact(args.synopses_artifact)
            synopses_artifact_path = synopses_artifact.file()
            synopses_df = pd.read_csv(synopses_artifact_path, low_memory=False)
            synopses_df.to_csv(synopses_file)

            all_anime_artifact = run.use_artifact(args.all_anime_artifact)
            all_anime_path = all_anime_artifact.file()
            all_anime_df = pd.read_csv(all_anime_path, low_memory=False)
            all_anime_df.to_csv(all_anime_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download files and upload them as artifacts to W&B",
        fromfile_prefix_chars="@")

    parser.add_argument(
        "--stats_url",
        type=str,
        help="File URL",
        required=True
    )

    parser.add_argument(
        "--stats_artifact",
        type=str,
        help="Name for artifact",
        required=True
    )

    parser.add_argument(
        "--stats_artifact_type",
        type=str,
        help="Type",
        required=True
    )

    parser.add_argument(
        "--stats_artifact_description",
        type=str,
        help="Description for the new artifact",
        required=True
    )

    parser.add_argument(
        "--synopses_url",
        type=str,
        help="File URL",
        required=True
    )

    parser.add_argument(
        "--synopses_artifact",
        type=str,
        help="Name for artifact",
        required=True
    )

    parser.add_argument(
        "--synopses_artifact_type",
        type=str,
        help="Type",
        required=True
    )

    parser.add_argument(
        "--synopses_artifact_description",
        type=str,
        help="Description for the new artifact",
        required=True
    )

    parser.add_argument(
        "--all_anime_url",
        type=str,
        help="File URL",
        required=True
    )

    parser.add_argument(
        "--all_anime_artifact",
        type=str,
        help="Name for artifact",
        required=True
    )

    parser.add_argument(
        "--all_anime_type",
        type=str,
        help="Type",
        required=True
    )

    parser.add_argument(
        "--all_anime_description",
        type=str,
        help="Description for the new artifact",
        required=True
    )

    parser.add_argument(
        "--save_raw_locally",
        type=lambda x: bool(strtobool(x)),
        help='Choose whether or not to save clean data frame to local file',
        required=True
    )

    args = parser.parse_args()
    go(args)
