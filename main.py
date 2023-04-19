import mlflow
import os
import hydra
import json
from omegaconf import DictConfig, OmegaConf


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

    root_path = hydra.utils.get_original_cwd()
    if isinstance(config["main"]["execute_steps"], str):
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        steps_to_execute = config["main"]["execute_steps"]