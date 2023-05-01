import argparse
import logging
# import os
import wandb
from distutils.util import strtobool
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import tensorflow.keras.layers as tfkl
import tensorflow.keras.callbacks as tfkc
import matplotlib.pyplot as plt


# model = tf.keras.models.load_model('./anime_nn.h5')
# weight = model.load_weights('./anime_weights.h5')

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def get_main_df():

    #run = wandb.init(project=args.project_name)
    #logger.info("Downloading data artifact")
    #artifact = run.use_artifact(args.input_data, type='preprocessed_data')
    #artifact_path = artifact.file()
    #df = pd.read_parquet(artifact_path)
    main_df = pd.read_parquet('/Applications/python_files/anime_recommendations/data/preprocessed_stats_full.parquet')
    print (main_df.shape)
    #anime_df = pd.read_csv('/Applications/python_files/anime_recommendations/Archive/nime.csv', low_memory=True)
    #snyopses_df = pd.read_csv('/Applications/python_files/anime_recommendations/data/synopses.csv', low_memory=True)

    # Encoding categorical data
    # Get all user ids and anime ids
    anime_ids = main_df["anime_id"].unique().tolist()

    # Dict of format {anime_id: count_number}
    anime_to_index = {value: count for count, value in enumerate(anime_ids)}
    index_to_anime = {count: value for count, value in enumerate(anime_ids)}
    main_df["anime"] = main_df["anime_id"].map(anime_to_index)
    main_df = main_df.drop(
    	['watching_status', 'watched_episodes', 'max_eps', 'half_eps'], axis=1)
    main_df = main_df.sample(frac=1, random_state=42)

    return main_df, anime_to_index, index_to_anime

def get_anime_df():
	import chardet
	file = '/Applications/python_files/anime_recommendations/archive/anime.csv'
	with open(file, 'rb') as rawdata:
		result = chardet.detect(rawdata.read(100000))
		print (f'result is {result}')
	df = pd.read_csv(file)
		#names=['MAL_ID', "Name", "Score", "Genres", "English name", "Japanese name", "Type",
		#"Episodes", "Aired", "Premiered", "Producers", "Liscensors", "Studios", "Source",
		#"Duration", "Rating", "Ranked", "Popularity", "Members", "Favorites", "Watching",
		#"Completed", "On-Hold", "Dropped", "Plan to Watch", "Score-10", "Score-9", "Score-8",
		#"Score-7", "Score-6", "Score-5", "Score-4", "Score-3", "Score-2", "Score-1"],
		#encoding='latin1',
		#lineterminator='\n')
	df = df.replace("Unknown", np.nan)
	def anime_name(anime_id):
		try:
			name = df[df.anime_id == anime_id].eng_version.values[0]
			if name is np.nan:
				name = df[df.anime_id == anime_id].Name.values[0]
		except:
			print ("Bad name value")
		return name

	df['anime_id'] = df["MAL_ID"]
	df['eng_version'] = df['English name']
	df['japanese_name'] = df["Japanese name"]
	df['eng_version'] = df.anime_id.apply(lambda x: anime_name(x))
	df.sort_values(by=['Score'],
		inplace=True,
		ascending=False,
		kind='quicksort',
		na_position='last')
	df = df[["anime_id", "eng_version", "japanese_name", "Score",
	"Genres", "Episodes", "Type", "Premiered", 'Studios', "Source"]]
	return df

def get_model():
	model = tf.keras.models.load_model('./models/wandb_anime_nn.h5')
	return model


def get_weights():
	#anime_weights = pd.read_csv('/Applications/python_files/anime_recommendations/models/wandb_anime_weights.csv')
	#user_weights = pd.read_csv('/Applications/python_files/anime_recommendations/models/wandb_user_weights.csv')
	model = get_model()
	anime_weights = model.get_layer('anime_embedding')
	anime_weights = anime_weights.get_weights()[0]
	anime_weights = anime_weights / np.linalg.norm(anime_weights, axis = 1).reshape((-1, 1))

	user_weights = model.get_layer('user_embedding')
	user_weights = user_weights.get_weights()[0]
	user_weights = user_weights / np.linalg.norm(user_weights, axis=1).reshape((-1,1))

	return anime_weights, user_weights

def get_sypnopses_df():
	sypnopses_df = pd.read_csv('/Applications/python_files/anime_recommendations/data/synopses.csv')
	return sypnopses_df

def get_sypnopsis(anime, sypnopsis_df):
    if isinstance(anime, int):
        return sypnopsis_df[sypnopsis_df.MAL_ID == anime].sypnopsis.values[0]
    if isinstance(anime, str):
        return sypnopsis_df[ssypnopsis_df.Name == anime].sypnopsis.values[0]

def get_anime_frame(anime, df):
    if isinstance(anime, int):
        return df[df.anime_id == anime]
    if isinstance(anime, str):
        return df[df['eng_version'] == anime]

model = get_model()
anime_weights, user_weights = get_weights()
anime_df = get_anime_df()
#synopses_df = get_synopses_df()

print (f'all model weights shape is {[e.shape for e in model.get_weights()]}') # [(261918, 128), (17178, 128), (1, 1), (1,), (1,), (1,), (1,), (1,)]
print (f'anime weights shape is {anime_weights.shape}') # (17178, 128) ### ALL SAME SHAPE EVEN IF NOT NORMALIZED
print (f'user weights shape is {user_weights.shape}') # (261918, 128)
print ('shape of anime weights layer is {}'.format(model.get_layer('anime_embedding'))) # keras.layers.core.embedding.Embedding

pd.set_option("max_colwidth", None)
def find_similar_anime(name, n=5, return_dist=False, neg=False):
    main_df, anime_to_index, index_to_anime = get_main_df()
    anime_weights, user_weights = get_weights()
    weights = anime_weights
    anime_df = get_anime_df()
    sypnopsis_df = get_sypnopses_df()
    #try:
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
        
    rindex = df

    SimilarityArr = []

    for close in closest:
        decoded_id = index_to_anime.get(close)
        sypnopsis = get_sypnopsis(decoded_id, sypnopsis_df)
        anime_frame = get_anime_frame(decoded_id, anime_df)
            
        anime_name = anime_frame['eng_version'].values[0]
        genre = anime_frame['Genres'].values[0]
        japanese_name = anime_frame['japanese_name'].values[0]
        episodes = anime_frame['Episodes'].values[0]
        Type = anime_frame['Type'].values[0]
        premiered = anime_frame['Premiered'].values[0]
        studios = anime_frame['Studios'].values[0]

        similarity = dists[close]
        SimilarityArr.append({"anime_id": decoded_id, "Name": anime_name,
                                  "Similarity": similarity,"Genre": genre,
                                  'Sypnopsis': sypnopsis, "Japanese name": japanese_name,
                                  "Episodes": episodes, "Type": Type,
                                  "Premiered": premiered, "Studios": studios})
    Frame = pd.DataFrame(SimilarityArr).sort_values(by="Similarity", ascending=False)
    return Frame[Frame.anime_id != index].drop(['anime_id'], axis=1)

    #except:
    #    print('{}!, Not Found in Anime list'.format(name))             



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an anime recommendation neural network",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
    	"--main_data",
    	type=str,
    	help="Wandb artifact of main data frame",
    	required=True
    )

    parser.add_argument(
    	"--synopses",
    	type=str,
    	help="Wandb artifact of anime synopses",
    	required=True
    )

    parser.add_argument(
    	"--all_anime",
    	type=str,
    	help="Wandb artifact containing list of all anime",
    	requried=True
    )

    parser.add_argument(
    	"--anime_weights",
    	type=str,
    	help="Wandb artifact containing numpy array of anime weights",
    	required=True
    )

    parser.add_argument(
    	"--user_weights",
    	type=str,
    	help="Wandb artifact containing numpy array of user weights",
    	required=True
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
