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
    mlflow run https://github.com/DyRutter/rental_prices.git -v 1.0.3 -P
    hydra_options="data.sample='sample2.csv'"
    """
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # root_path = hydra.utils.get_original_cwd()
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
                "all_anime_artifact": config["data"]["synopses_artifact"],
                # "raw_data"
                "all_anime_type": config["data"]["all_anime_type"],
                # "Main statistics file"
                "all_anime_description": config["data"]
                    ["all_anime_description"]})


if __name__ == "__main__":
    go()
