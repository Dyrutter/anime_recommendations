import argparse
import logging
import os
import wandb
from distutils.util import strtobool
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ast


logging.basicConfig(
    filename='./similar_anime.log', #Path to log file
    level=logging.INFO, #Log info, warnings, errors, and critical errors
    filemode='a', #Create log file if one doesn't already exist and add 
    format='%(asctime)s-%(name)s - %(levelname)s - %(message)s',
    datefmt='%d %b %Y %H:%M:%S %Z', #Format date
    force=True)
logger = logging.getLogger()


def get_df():
    """
    Get data frame from wandb
    Covert to same format we used for neural network
    """
    run = wandb.init(project=args.project_name)
    logger.info("Downloading data artifact")
    artifact = run.use_artifact(args.main_df, type='preprocessed_data')
    artifact_path = artifact.file()
    df = pd.read_parquet(artifact_path)
    logger.info("Main preprocessed df shape is %s", df.shape)
    logger.info("Main preprocessed df head is %s", df.head())
    
    # Encoding categorical data
    user_ids = df["user_id"].unique().tolist()
    anime_ids = df["anime_id"].unique().tolist()

    # Dicts of format {id: count_number}
    anime_to_index = {value: count for count, value in enumerate(anime_ids)}
    user_to_index = {value: count for count, value in enumerate(user_ids)}

    # Dicts of format {count_number: id}
    index_to_user = {count: value for count, value in enumerate(user_ids)}
    index_to_anime = {count: value for count, value in enumerate(anime_ids)}

    # Convert values of format id to count_number
    df["user"] = df["user_id"].map(user_to_index)
    df["anime"] = df["anime_id"].map(anime_to_index)
    df = df[['user', 'anime', 'rating']]   
    df = df.sample(frac=1, random_state=42)

    logger.info("Final preprocessed df shape is %s", df.shape)
    logger.info("Final preprocessed df head is %s", df.head())

    return df, anime_to_index, index_to_anime


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
    return df

def update_anime_df():
    # Convert column spaces to underscores 
    df = get_anime_df()
    df['anime_id'] = df['MAL_ID']
    df['japanese_name'] = df['Japanese name']
    df["eng_version"] = df['English name']
    logger.info("Original English version is %s", df["eng_version"].head())
    df['eng_version'] = df.anime_id.apply(lambda x: anime_name(x))
    logger.info("New English version is %s", df["eng_version"].head())
    df.sort_values(by=['Score'], 
               inplace=True,
               ascending=False, 
               kind='quicksort',
               na_position='last')
    keep_cols = ["anime_id", "eng_version", "Score", "Genres", "Episodes",
        "Premiered", "Studios", "japanese_name"]
    df = df[keep_cols]
    logger.info("Final anime df head is %s", df.head())
    logger.info("Final anime df shape is %s", df.shape)
    return df

def anime_name(anime_id):
	df = get_anime_df()
    try:
        name = df[df.anime_id == anime_id].eng_version.values[0]
        if name is np.nan:
            name = df[df.anime_id == anime_id].Name.values[0]
    except:
        print('error')    
    return name

def getAnimeFrame(anime):
	df = update_anime_df()
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
    logger.info("Anime df head is %s", df.head())
    return df

def getSypnopsis(anime):
    if isinstance(anime, int):
        return sypnopsis_df[sypnopsis_df.MAL_ID == anime].sypnopsis.values[0]
    if isinstance(anime, str):
        return sypnopsis_df[sypnopsis_df.Name == anime].sypnopsis.values[0]

def get_model():
    run = wandb.init(project=args.project_name)
    logger.info("Downloading model")
    artifact = run.use_artifact(args.model, type='h5')
    artifact_path = artifact.file()
    model = tf.keras.models.load_model(artifact_path)
    return model

def get_weights():
    run = wandb.init(project=args.project_name)
    logger.info("Getting weights")
    model = get_model()
    anime_weights = model.get_layer('anime_embedding')
    anime_weights = anime_weights.get_weights()[0]
    anime_weights = anime_weights / np.linalg.norm(
    	anime_weights, axis = 1).reshape((-1, 1))

    user_weights = model.get_layer('user_embedding')
    user_weights = user_weights.get_weights()[0]
    user_weights = user_weights / np.linalg.norm(
    	user_weights, axis=1).reshape((-1,1))
    logger.info("Weights extracted!")
    return anime_weights, user_weights

pd.set_option("max_colwidth", None)

def find_similar_anime(name, n=5, return_dist=False, neg=False):
    df, anime_to_index, index_to_anime = get_df()
    anime_weights, user_weights = get_weights()
    weights = anime_weights
    anime_df = get_anime_df()
    sypnopsis_df = get_sypnopses_df()
    try:
        index = get_anime_frame(name, anime_df).anime_id.values[0]
        encoded_index = anime_to_index.get(index)
        
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)
        
        n = n + 1            
        
        if neg:
            closest = sorted_dists[:n]
        else:
            closest = sorted_dists[-n:]

        print('animes closest to {}'.format(name))

        if return_dist:
            return dists, closest
        
        rindex = anime_df

        SimilarityArr = []

        for close in closest:
            decoded_id = index_to_anime.get(close)
            sypnopsis = get_sypnopsis(decoded_id, sypnopsis_df)
            anime_frame = get_anime_frame(decoded_id, anime_df)
            
            anime_name = anime_frame['eng_version'].values[0]
            genre = anime_frame['Genres'].values[0]
            japanese_name = anime_frame['japanese_name'].values[0]
            episodes = anime_frame['Episodes'].values[0]
            premiered = anime_frame['Premiered'].values[0]
            studios = anime_frame['Studios'].values[0]

            similarity = dists[close]
            SimilarityArr.append(
                {"anime_id": decoded_id, "Name": anime_name,
                "Similarity": similarity, "Genre": genre,
                'Sypnopsis': sypnopsis, "Episodes": episodes,
                "Japanese name": japanese_name, "Studios": stuidos,
                "Premiered": premiered})

        Frame = pd.DataFrame(SimilarityArr).sort_values(
        	by="Similarity", ascending=False)
        return Frame[Frame.anime_id != index].drop(['anime_id'], axis=1)
    except:
    	print (f'{name} not found in data base!')


with open(ingestedfiles, 'r') as fp:
    dataset_list = ast.literal_eval(fp.read())

def go(args):
    run = wandb.init(
        project=args.project_name,
        name="similar_anime")
    rating_df, anime_to_index, index_to_anime = get_df()
    anime_df = get_anime_df()
    sypnoses_df = get_sypnopses_df()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an anime recommendation neural network",
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

    args = parser.parse_args()
    go(args)
