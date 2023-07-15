# Anime Recommendations

The scripts in this project can be used to find anime recommendations. Three different systems were created. The first recommends anime based on similar anime. The second recommends anime based on the preferences of similar users. The third recommends anime based on ratings a user has assigned to previously-watched anime. MLOps are incorporated using MLflow and Weights & Biases, allowing the three systems to be run simultaneously.
The workflow components are:

1) Download: Download raw data and save it as wandb artifacts 
2) Preprocess: Clean and engineer the data and save it as wandb artifacts
3) Neural Network: Create and train an embedding-based neural network and save it as a wandb artifact
4) Similar Anime: Recommend anime according to similar anime (determined by weight cosine similarity)
5) Similar Users: Find users similar to an input user (determined by weight cosine similarity)
6) User Prefs: Find an input user's preferred genres and sources based on shows they've previously rated highly
7) User Recs: Recommend anime based on similar users' preferences
8) Model Recs: Recommend anime based on neural network model rankings (determined by collaborative filtering)

# ENVIRONMENT SETUP

## PREREQUISITES

+ Weights and Biases account, which can be created [here](https://wandb.ai/site)
+ GitHub account (for running)
+ Clone GitHub repo `https://github.com/Dyrutter/rental_prices.git`
+ A python 3.10 `conda` virtual environment

## DEPENDENCIES

Install requirements found in [requirements.txt file](./requirements.txt)
 
Due to the size of the data set (2.1 gb), the model was trained using [Tensor Processing Units/TPUs](https://www.tensorflow.org/guide/tpu), Google's application-specific integrated circuits (ASICs) which drastically accelerate machine learning workloads. If you intend to run the project using a TPU inside a Colab notebook as opposed to on your local machine, make sure that your hardware accelerator is set to TPU by checking your notebook settings: 

`Runtime > Change runtime type > Hardware accelerator > TPU`

Note: If using M1 or M2 Mac, install tensorflow with the commands:

`SYSTEM_VERSION_COMPAT=0 python -m pip install tensorflow-macos`

`SYSTEM_VERSION_COMPAT=0 python -m pip install tensorflow-metal`

# THE DATA

The data were scraped from [myanimelist.net](https://myanimelist.net), an online database containing over 18,000 anime and 350,000 users. The size of the main data set in csv format is 2.03 GB, too large to store in github. For the sake of casual reproducibility, I reduced the data set from 109 million samples to roughly seven million samples, taking care to ensure all anime remained included, and stored it in parquet format [here](https://github.com/Dyrutter/anime_recommendations/blob/main/data/user_stats.parquet).     

The [main data set, user_stats.parquet](https://github.com/Dyrutter/anime_recommendations/blob/main/data/user_stats.parquet) contains information about every anime and every user. It is formatted as such:

![](https://github.com/Dyrutter/anime_recommendations/blob/main/data/main_data_set_example.png)

Where "watching status" includes
+ Currently watching (1)
+ Completed (2)
+ On hold (3)
+ Dropped (4)
+ Plan to watch (6)
 

[all_anime.csv](https://github.com/Dyrutter/anime_recommendations/blob/main/data/all_anime.csv) contains general information of every anime including genre, average score, English name, Japanese name, type (TV, movie, OVA, Special, or ONA), Number of episodes, Broadcast date, Season premier, producers, liscensors, studios, source (Manga, light novel, book, original), episode duration, age rating, rank, popularity, members, number of favorites, watching statuses, and scores.

Note: The data were collected in early 2023. Since then, myanimelist.net has altered its policy such that it is no longer possible to scrape data using traditional methods. They instead produced [their own API](https://myanimelist.net/apiconfig) for accessing data. The API is (currently) in its early stages of development and is not conduit to scraping large amounts of data. As such, it is very difficult to recreate the datasets used in this project.

# MLFLOW COMPONENTS:

[run.py (segregate component)](./segregate/run.py)

### [Download](./download/download.py) 
+ Downloads data sets from url specified in [the hydra config file](./config/config.yaml)
+ Converts data sets into Weights & Biases artifacts
+ Uploads the artifacts to Weights & Biases
    
### [Preprocess](./preprocess/preprocess.py) 
+ Scales ratings data and performs basic cleaning
+ Drops users who have watched fewer than a certain number of anime as specified in [the hydra config file](./config/config.yaml)
+ If desired, drops instances in which a user has not watched but plans to watch a given anime
+ If desired, drops instances in which a user has watched fewer than half the total episodes of an anime
+ Converts data sets into artifacts and uploads them to Weights & Biases
   
### [Neural Network](./neural_network/neural_network.py)
+ Create and trains an embedding-based neural network using tensor flow
+ Produces and saves an .h5 file of the trained model, an .h5 file of the model's weights, and a csv file of the model's history
+ Converts model files into artifacts and uploads them to Weights & Biases
+ [The hydra config file](./config/config.yaml) contains several options for altering the model's structure during compilation

### [Similar Anime](./similar_anime/similar_anime.py)
+ Downloads model artifacts and data set artifacts from Weights & Biases
+ Extracts weights and computes cosine similarity to recommend anime according to similar anime
+ If designated as such in [the hydra config file](./config/config.yaml), recommendations only include anime of specified genres and media types
+ Can recommend based on either a specified anime or a random anime
+ Creates a csv file of recommendations and uploads it as an artifact to Weights & Biases
   
### [Similar Users](./similar_users/similar_users.py)
+ Extracts weights from the wandb model artifact and computes cosine similarity to find a specified number users similar to an input User ID
+ The input user ID can be either designated in [the hydra config file](./config/config.yaml) file or chosen randomly
+ Creats a csv file specifying the assessed user ID and another csv file of similar users and uploads the files as artifacts to Weights & Biases

### [User Prefs](./user_prefs/user_prefs.py) 
+ Find a user's preferred genres and sources based on shows they've rated highly
+ Creates word cloud images of the user's preferred genres and sources
+ Uploads preferences csv file and word clouds to Weights & Biases as artifacts
+ Input user can be either taken from the similar users artifact, newly specified in the config file, or a random user
   
### [User Recs](./user_recs/user_recs.py) 
+ Recommend anime based on similar users' preferences
+ Input user can be either taken from the similar users artifact, newly specified in the config file, or a random user
+ If the ID artifact from the similar users component is used, the corresponding similar users artifact and user preferences artifact are also used. Otherwise, new sets of similar users and user preferences are computed
+ Can restrict the anime recommendations to include only certain genres if desired
+ Recommendations are ranked according to the number of similar users who favorited the same anime (e.g. if 10 out of 10 similar users favorited Sword Art Online, Sword Art Online would be rated at the top of the list in conjunction with any other anime 10 out of 10 similar users had favorited)
+ Uploads recommendations csv file artifact to Weights & Biases as well as a preferences csv file and favorite genres and sources word clouds

### [Model Recs](./model_recs/model_recs.py) 
+ Recommend animes based on the neural network model's predicted ratings of unwatched animes
+ Input user can be either taken from the similar users artifact, newly specified in the config file, or a random user
+ Can restrict the anime recommendations to include only certain genres and/or media types (e.g. Movies, TV shows, OVAs etc.) if desired
+ Produces csv file of a specified number of anime recommendations and uploads it as an artifact to Weights & Biases

## ROOT FILES

### [main.py (Root Directory)](./main.py)
+ Defines each MLFLow component
+ Specifies hyperparameters for each component using argparse in conjunction with [Hydra](https://hydra.cc/docs/intro/)
+ Specifies input and output Weights & Biases artifacts
+ From root directory, the workflow can be run locally with command `mlflow run .`
+ Specific components (e.g. "download") can be run locally with `mlflow run . -P hydra_options="main.execute_steps='download'"`
+ Can be run on GitHub using the command `mlflow run https://github.com/Dyrutter/anime_recommendations.git -v 1.0.2`

### [config.yaml](./config/config.yaml)
+ Hydra configuration file containing settings MLflow runs
+ Contains modifiable customization options for each MLflow component, nearly 200 in total

## OTHER FILES
+ conda.yaml dependencies files exist in each component and in the main directory
+ MLproject configuration files exist in each component and in the main directory
