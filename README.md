# Anime Recommendation Algorithms

MLOps are incorporated using MLflow, Weights and Biases (wandb), and FastAPI. 
The workflow:

1) Download raw data and save it as wandb artifacts 
2) Clean and engineer the data and save it as wandb artifacts
3) Create and train an embedding-based neural network and save it as a wandb artifact
4) Recommend anime according to similar anime
5) Find users similar to an input user
6) Find an input user's preferred genres and sources
7) Recommend anime based on similar users' preferences
8) Recommend anime based on neural network model rankings


Due to the size of the data set, the model was trained using [Tensor Processing Units/TPUs](https://www.tensorflow.org/guide/tpu), Google's application-specific integrated circuits (ASICs) which drastically accelerate machine learning workloads. If you intend to run the project using a TPU inside a Colab notebook as opposed to on your local machine, make sure that your hardware accelerator is set to TPU by checking your notebook settings: 

`Runtime > Change runtime type > Hardware accelerator > TPU`

Note: If using M1 or M2 Mac, install tensorflow with the commands:

`SYSTEM_VERSION_COMPAT=0 python -m pip install tensorflow-macos`

`SYSTEM_VERSION_COMPAT=0 python -m pip install tensorflow-metal`

# THE DATA

The data were scraped from [myanimelist.net](https://myanimelist.net), an online database containing over 18,000 anime and 350,000 users. The size of the main data set in csv format is 2.03 GB, far too large to store in github. For the sake of casual reproducibility, I reduced the data set from 109 million samples to roughly seven million samples, taking care to ensure all anime remained included, and stored it in parquet format [here](https://github.com/Dyrutter/anime_recommendations/blob/main/data/user_stats.parquet).     

The [main data set, user_stats.parquet](https://github.com/Dyrutter/anime_recommendations/blob/main/data/user_stats.parquet) contains information about every anime and every user. It is formatted as such:

![](https://github.com/Dyrutter/anime_recommendations/blob/main/data/main_data_set_example.png)

Where "watching status" includes
+ Currently watching (1)
+ Completed (2)
+ On hold (3)
+ Dropped (4)
+ Plan to watch (6)
 

[all_anime.csv](https://github.com/Dyrutter/anime_recommendations/blob/main/data/all_anime.csv) scontain general information of every anime including genre, average score, English name, Japanese name, type (TV, movie, OVA, Special, or ONA), Number of episodes, Broadcast date, Season premier, producers, liscensors, studios, source (Manga, light novel, book, original), episode duration, age rating, rank, popularity, members, number of favorites, watching statuses, and scores.

Special thanks to:
myanimelist.net for hosting the data
