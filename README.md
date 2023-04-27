# Creating an Anime Recommendation System

This is an example of creating anime an recommendation system. The workflow is:

1) Download raw data
2) Clean and engineer data
3) Create and train an embedding-based neural network
4) Recommend anime according to similar anime
5) Recommend anime according to similar users
6) Recommend anime for specific users
7) Create an API for live recommendations

MLOps are incorporated using MLflow, Weights and Biases, and FastAPI.

Due to the large amount of data, I train the model using [Tensor Processing Units/TPUs](https://www.tensorflow.org/guide/tpu), Google's application-specific integrated circuits (ASICs) which drastically accelerate machine learning workloads. Before you run a Colab notebook, make sure that your hardware accelerator is a TPU by checking your notebook settings: 

`Runtime > Change runtime type > Hardware accelerator > TPU`

Note: If using M1 or M2 Mac, install tensorflow with the commands:

`SYSTEM_VERSION_COMPAT=0 python -m pip install tensorflow-macos`

`SYSTEM_VERSION_COMPAT=0 python -m pip install tensorflow-metal`

