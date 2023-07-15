# Anime Recommendations

The scripts in this project can be used to find anime recommendations. MLOps are incorporated using MLflow, Weights and Biases (wandb), and FastAPI. 
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

# PRIMARY FILES

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

### OTHER FILES
+ A conda.yaml dependencies file exists in each component for use by MLFlow
+ An MLproject configuration file exists in each component for use by MLFLow
