# ANIME RECOMMENDATION SYSTEMS

The Python scripts in this project return anime recommendations. Three different systems were created. The first recommends animes based on similar animes. The second recommends anime based on the preferences of similar users. The third recommends anime based on the ratings a user has assigned to previously watched anime. MLOps are incorporated using MLflow and Weights & Biases, allowing the three systems to be run simultaneously.

The MLflow components are:

1) Download: Download raw data to local machine from GitHub and save as WandB artifacts
2) Preprocess: Clean and engineer the data and save as WandbB artifacts
3) Neural Network: Create and train an embedding-based neural network
4) Similar Anime: Recommend anime according to similar anime (determined by weight cosine similarity)
5) Similar Users: Find users similar to an input user (determined by weight cosine similarity)
6) User Prefs: Find an input user's preferred genres and sources based on shows they've previously watched
7) User Recs: Recommend anime based on the preferences of similar users
8) Model Recs: Recommend anime based on the neural network model's predicted anime ratings

Further details regarding how the models were constructed can be found in the [model card](./model_card.md).
Individual functions and code methodologies are further detailed in each component's python script. 

# ENVIRONMENT SETUP

### PREREQUISITES

+ Weights and Biases account, which can be created [at their website](https://wandb.ai/site)
+ GitHub account (for running)
+ Clone GitHub repo `https://github.com/Dyrutter/anime_recommendations.git`
+ A python 3.10 `conda` virtual environment

### DEPENDENCIES

Necessary installation libraries are listed in [requirements.txt file](./requirements.txt)
 
Due to the size of the data set (2.1 gb), the model was trained using [Tensor Processing Units/TPUs](https://www.tensorflow.org/guide/tpu), Google's application-specific integrated circuits (ASICs) which drastically accelerate machine learning workloads. If you intend to run the project using a TPU inside a Colab notebook as opposed to on your local machine, make sure that your hardware accelerator is set to TPU by checking your notebook settings: 

`Runtime > Change runtime type > Hardware accelerator > TPU`
It is also necessary to set "TPU_init" in the [config yaml file](./config/config.yaml) to True

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

### [DOWNLOAD](./download/download.py) 
+ Downloads data sets from url specified in [the hydra config file](./config/config.yaml)
+ Converts data sets into Weights & Biases artifacts
+ Uploads the artifacts to Weights & Biases
    
### [PREPROCESS](./preprocess/preprocess.py) 
+ Scales ratings data and performs basic cleaning
+ Drops users who have watched fewer than a certain number of anime as specified in the hydra config file
+ If desired, drops instances in which a user has not watched but plans to watch a given anime
+ If desired, drops instances in which a user has watched fewer than half the total episodes of an anime
+ Converts data sets into artifacts and uploads them to Weights & Biases
   
### [NEURAL NETWORK](./neural_network/neural_network.py)
+ Create and trains an embedding-based neural network using tensor flow
+ Produces and saves an .h5 file of the trained model, an .h5 file of the model's weights, and a csv file of the model's history
+ Converts model files into artifacts and uploads them to Weights & Biases
+ The hydra config file contains several options for altering the model's structure during compilation

Run at 20 epochs, the model takes roughly two hours to train on an M1 Mac. If TPU option is enabled, it can be completed in under 15 minutes.
The change in loss is visualized below.

![](https://github.com/Dyrutter/anime_recommendations/blob/main/figure_file/Neural_Network_Loss.png)
#### Output graph of neural network's loss per epoch history

An optimum Mean Squared Error (MSE) of 0.05074 and validation MSE of 0.07199 was reached during epoch 14, where the learning rate was 1.5368E-05.
The full history can be seen [here](https://github.com/Dyrutter/anime_recommendations/blob/main/figure_file/anime_nn_history.csv).

### [SIMILAR ANIME](./similar_anime/similar_anime.py)
+ Downloads model artifacts and data set artifacts from Weights & Biases
+ Extracts and analyzes weights to recommend animes based on their similarity to an input anime
+ If designated as such in the hydra config file, recommendations only include anime of specified genres and media types
+ Can recommend based on either a specified anime or a random anime
+ Creates a csv file of recommendations and uploads it as an artifact to Weights & Biases

Likeness between animes was ascertained by comparing the cosine similarites between the input anime's embedding vector weights and the embedding vector weights of every other anime. The process is explained in more detail in the [model card](https://github.com/Dyrutter/anime_recommendations/blob/main/model_card.md).  
   
### [SIMILAR USERS](./similar_users/similar_users.py)
+ Extracts weights from the wandb model artifact and computes cosine similarity to find a specified number users similar to an input User ID
+ The input user ID can be either designated in the hydra config file file or chosen randomly
+ Creats a csv file specifying the assessed user ID and another csv file of similar users and uploads the files as artifacts to Weights & Biases

Likeness between users was ascertained by comparing the cosine similarites between the input user's embedding vector weights and the embedding vector weights of every other user. The process is explained in more detail in the [model card](https://github.com/Dyrutter/anime_recommendations/blob/main/model_card.md).

### [USER PREFERENCES](./user_prefs/user_prefs.py) 
+ Find a user's preferred genres and sources based on shows they've rated highly
+ Creates word cloud images of the user's preferred genres and sources
+ Uploads preferences csv file and word clouds to Weights & Biases as artifacts
+ Input user can be either taken from the similar users artifact, newly specified in the config file, or a random user

![](https://github.com/Dyrutter/anime_recommendations/blob/main/figure_file/User_ID_153695_favorite_genres.png)

#### Example of a genre word cloud. Word sizes are determined according to the number of animes of each given genre a user has watched, with more shows watched corresponding to larger texts 
   
### [SIMILAR USER BASED RECOMMENDATIONS](./user_recs/user_recs.py) 
+ Recommend anime based on similar users' preferences
+ Input user can be either taken from the similar users artifact, newly specified in the config file, or a random user
+ If the ID artifact from the similar users component is used, the corresponding similar users artifact and user preferences artifact are also used. Otherwise, new sets of similar users and user preferences are computed
+ Can restrict the anime recommendations to include only certain genres if desired
+ Recommendations are ranked according to the number of similar users who favorited the same anime (e.g. if 10 out of 10 similar users favorited Sword Art Online, Sword Art Online would be rated at the top of the list in conjunction with any other anime 10 out of 10 similar users had favorited)
+ Uploads recommendations csv file artifact to Weights & Biases as well as a preferences csv file and favorite genres and sources word clouds

### [MODEL RATING RECOMMENDATIONS](./model_recs/model_recs.py) 
+ Recommend animes based on the neural network model's predicted ratings of unwatched animes
+ Input user can be either taken from the similar users artifact, newly specified in the config file, or a random user
+ Can restrict the anime recommendations to include only certain genres and/or media types (e.g. Movies, TV shows, OVAs etc.) if desired
+ Produces a csv file of a specified number of anime recommendations and uploads it as an artifact to Weights & Biases

## SUPPORTING FILES

### [main.py](./main.py)
+ Defines each MLFLow component
+ Specifies hyperparameters for each component using argparse in conjunction with [Hydra](https://hydra.cc/docs/intro/)
+ Specifies input and output Weights & Biases artifacts
+ From root directory, the workflow can be run locally with command `mlflow run .`
+ Specific components (e.g. "download") can be run locally with `mlflow run . -P hydra_options="main.execute_steps='download'"`
+ Can be run on GitHub using the command `mlflow run https://github.com/Dyrutter/anime_recommendations.git -v 1.0.0`

### [config.yaml](./config/config.yaml)
+ Hydra configuration file containing settings for MLflow runs
+ Contains modifiable customization options for each MLflow component, nearly 200 in total

### [conda.yml](./conda.yml) 
+ Container files for establishing MLFlow component environments
+ A separate, unique file is included in each component and in the root directory

### [MLproject](./MLproject)
+ MLFlow files used in conjunction with the hydra config file and python script argparse settings to manage customization options 
+ A separate, unique file is inlcuded in each component and in the main directory

### [Figure File](./figure_file)
+ Contains examples of all png images produced during each full MLflow run
+ Contains examples of all csv files produced during each full MLflow run
+ Contains figures used in the readme and model card

### [model_card.md](./model_card.md)
+ Explains collaborative filtering, cosine similarity, and other processes through which the recommendation systems were developed
+ Describes the final neural network model and its settings
+ Discusses the model's creation, usage, and limitations

## IDEAS FOR IMPROVEMENT
+ Create a program for re-creating and updating this data set using myanimelist.net's new API
+ Create a front-end API for accessing the recommendation systems
+ Re-train the model using alternate weight-initialization methods and compare for best results
+ Use a deep neural network with multiple hidden layers

