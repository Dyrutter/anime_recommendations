import mlflow
import os
import hydra
# import json
from omegaconf import DictConfig  # , OmegaConf


@hydra.main(config_name='config', config_path='config', version_base='2.5')
def go(config: DictConfig):
    """
    Run MLflow project. From main directory, download step can be run using:
    mlflow run . -P hydra_options="main.execute_steps='download'"
    Can be run in github using:
    mlflow run https://github.com/DyRutter/anime_recommendations.git -v 1.X -P
    hydra_options="data.sample='sampleX.csv'"
    """
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["ANIME_PROJECT"] = config["main"]["project_name"]
    os.environ["ANIME_GROUP"] = config["main"]["experiment_name"]

    root_path = hydra.utils.get_original_cwd()
    if isinstance(config["main"]["execute_steps"], str):
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        steps_to_execute = config["main"]["execute_steps"]

    if "download" in steps_to_execute:
        _ = mlflow.run(
            f"{config['main']['download_repository']}",
            version="main",
            entry_point="main",
            parameters={
                # Name of project
                "project_name": config["main"]["project_name"],
                # URL to stats data sample
                "stats_url": config["data"]["stats_url"],
                # "user_stats.parquet",
                "stats_artifact": config["data"]["stats_artifact"],
                # "raw_data"
                "stats_artifact_type": config["data"]["stats_artifact_type"],
                # "Main statistics file"
                "stats_artifact_description": config["data"]
                    ["stats_artifact_description"],
                # URL to synopses data sample
                "synopses_url": config["data"]["synopses_url"],
                # "synopses.csv",
                "synopses_artifact": config["data"]["synopses_artifact"],
                # "raw_data"
                "synopses_artifact_type": config["data"]
                    ["synopses_artifact_type"],
                # "Main synopses file"
                "synopses_artifact_description": config["data"]
                    ["synopses_artifact_description"],
                "all_anime_url": config["data"]["all_anime_url"],
                # "all_anime.csv"
                "all_anime_artifact": config["data"]["all_anime_artifact"],
                # "raw_data"
                "all_anime_type": config["data"]["all_anime_type"],
                # "Main statistics file"
                "all_anime_description": config["data"]
                    ["all_anime_description"],
                "save_raw_locally": config["data"]["save_raw_locally"],
                "from_local": config["data"]["from_local"],
                "local_fname": config['data']['local_fname']})

    if "preprocess" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "preprocess"),
            entry_point="main",
            parameters={
                "raw_stats": config["data"]["stats_artifact_latest"],
                "project_name": config["main"]["project_name"],
                "preprocessed_stats": config["data"]["preprocessed_stats"],
                "preprocessed_artifact_type": config["data"]
                ["preprocessed_artifact_type"],
                "preprocessed_artifact_description": config["data"]
                ["preprocessed_artifact_description"],
                "num_reviews": config["data"]["num_reviews"],
                "drop_half_watched": config["data"]["drop_half_watched"],
                "save_clean_locally": config["data"]["save_clean_locally"],
                "drop_plan": config["data"]["drop_plan"],
                "drop_unwatched": config["data"]["drop_unwatched"]})

    if "neural_network" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "neural_network"),
            entry_point="main",
            parameters={
                "test_size": config["model"]["test_size"],
                "TPU_INIT": config["model"]["TPU_INIT"],
                "embedding_size": config["model"]["embedding_size"],
                "kernel_initializer": config["model"]["kernel_initializer"],
                "activation_function": config["model"]["activation_function"],
                "model_loss": config["model"]["model_loss"],
                "optimizer": config["model"]["optimizer"],
                "start_lr": config["model"]["start_lr"],
                "min_lr": config["model"]["min_lr"],
                "max_lr": config["model"]["max_lr"],
                "batch_size": config["model"]["batch_size"],
                "rampup_epochs": config["model"]["rampup_epochs"],
                "sustain_epochs": config["model"]["sustain_epochs"],
                "exp_decay": config["model"]["exp_decay"],
                "weights_artifact": config["model"]["weights_artifact"],
                "save_weights_only": config["model"]["save_weights_only"],
                "checkpoint_metric": config["model"]["checkpoint_metric"],
                "save_freq": config["model"]["save_freq"],
                "save_best_weights": config["model"]["save_best_weights"],
                "mode": config["model"]["mode"],
                "verbose": config["model"]["verbose"],
                "epochs": config["model"]["epochs"],
                "save_model": config["model"]["save_model"],
                "model_name": config["model"]["model_name"],
                "input_data": config["data"]["preprocessed_artifact_latest"],
                "project_name": config["main"]["project_name"],
                "model_artifact": config["model"]["model_artifact"],
                "history_csv": config["model"]["history_csv"]})

    if "similar_anime" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "similar_anime"),
            entry_point="main",
            parameters={
                "weights": config["nn_arts"]["main_weights_art"],
                "history": config["nn_arts"]["main_history"],
                "model": config["nn_arts"]["main_model"],
                "project_name": config["main"]["project_name"],
                "main_df": config["data"]["preprocessed_artifact_latest"],
                "sypnopses_df": config["data"]["sypnopses_artifact_latest"],
                "anime_df": config["data"]["all_anime_artifact_latest"],
                "anime_query": config["similarity"]["anime_query"],
                "a_query_number": config["similarity"]["a_query_number"],
                "random_anime": config["similarity"]["random_anime"]})

    if "similar_users" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "similar_users"),
            entry_point="main",
            parameters={
                "weights": config["nn_arts"]["main_weights_art"],
                "history": config["nn_arts"]["main_history"],
                "model": config["nn_arts"]["main_model"],
                "project_name": config["main"]["project_name"],
                "main_df": config["data"]["preprocessed_artifact_latest"],
                "user_query": config["similarity"]["user_query"],
                "id_query_number": config["similarity"]["id_query_number"],
                "max_ratings": config["similarity"]["max_ratings"],
                "random_user": config["similarity"]["random_user"]})

    if "recommend" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "recommend"),
            entry_point="main",
            parameters={
                "weights": config["nn_arts"]["main_weights_art"],
                "history": config["nn_arts"]["main_history"],
                "model": config["nn_arts"]["main_model"],
                "project_name": config["main"]["project_name"],
                "main_df": config["data"]["preprocessed_artifact_latest"],
                "sypnopses_df": config["data"]["sypnopses_artifact_latest"],
                "anime_df": config["data"]["all_anime_artifact_latest"],
                "anime_query": config["similarity"]["anime_query"],
                "a_query_number": config["similarity"]["a_query_number"]})


if __name__ == "__main__":
    go()
