# Creating an anime recommendation system
This is a workflow for creating anime an recommendation system that predicts what users will enjoy based on their similarities to other users. MLOps are incorporated using MLflow, Weights and Biases, and FastAPI.
Due to the large amount of data, I train the model using [Tensor Processing Units/TPUs](https://www.tensorflow.org/guide/tpu), Google's application-specific integrated circuits (ASICs) which drastically accelerate machine learning workloads.
Note: If using M1 Mac: nstall tensorflow with command: SYSTEM_VERSION_COMPAT=0 python -m pip install tensorflow-macos

