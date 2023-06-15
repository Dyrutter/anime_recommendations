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
                "history_csv": config["model"]["history_csv"],
                "ID_emb_name": config["model"]["ID_emb_name"],
                "anime_emb_name": config["model"]["anime_emb_name"],
                "merged_name": config["model"]["merged_name"],
                "main_df_type": config["data"]["preprocessed_artifact_type"],
                "model_type": config["nn_arts"]["model_type"],
                "history_type": config["nn_arts"]["history_type"],
                "weights_type": config["nn_arts"]["weights_type"],
                "model_metrics": config["model"]["model_metrics"]})

    if "similar_anime" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "similar_anime"),
            entry_point="main",
            parameters={
                "anime_df_type": config["data"]["all_anime_type"],
                "main_df_type": config["data"]["preprocessed_artifact_type"],
                "sypnopsis_df_type": config["data"]["synopses_artifact_type"],
                "model_type": config["nn_arts"]["model_type"],
                "model": config["nn_arts"]["main_model"],
                "project_name": config["main"]["project_name"],
                "main_df": config["data"]["preprocessed_artifact_latest"],
                "sypnopses_df": config["data"]["sypnopses_artifact_latest"],
                "anime_df": config["data"]["all_anime_artifact_latest"],
                "anime_query": config["similarity"]["anime_query"],
                "a_query_number": config["similarity"]["a_query_number"],
                "random_anime": config["similarity"]["random_anime"],
                "anime_rec_genres": config["similarity"]["anime_rec_genres"],
                "an_spec_genres": config["similarity"]["an_spec_genres"],
                "types": config["similarity"]["types"],
                "spec_types": config["similarity"]["spec_types"],
                "a_rec_type": config["similarity"]["a_rec_type"],
                "save_sim_anime": config["similarity"]["save_sim_anime"],
                "ID_emb_name": config["model"]["ID_emb_name"],
                "anime_emb_name": config["model"]["anime_emb_name"]})

    if "similar_users" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "similar_users"),
            entry_point="main",
            parameters={
                "anime_df": config["data"]["all_anime_artifact_latest"],
                "anime_df_type": config["data"]["all_anime_type"],
                "model": config["nn_arts"]["main_model"],
                "model_type": config["nn_arts"]["model_type"],
                "project_name": config["main"]["project_name"],
                "main_df": config["data"]["preprocessed_artifact_latest"],
                "main_df_type": config["data"]["preprocessed_artifact_type"],
                "sim_user_query": config["users"]["sim_user_query"],
                "id_query_number": config["similarity"]["id_query_number"],
                "max_ratings": config["similarity"]["max_ratings"],
                "sim_random_user": config["similarity"]["sim_random_user"],
                "num_faves": config["users"]["num_faves"],
                "TV_only": config["users"]["TV_only"],
                "sim_users_fn": config["users"]["sim_users_fn"],
                "sim_users_type": config["users"]["sim_users_type"],
                "ID_fn": config["users"]["ID_fn"],
                "ID_type": config["users"]["ID_type"],
                "ID_emb_name": config["model"]["ID_emb_name"],
                "anime_emb_name": config["model"]["anime_emb_name"],
                "save_sim_locally": config["users"]["save_sim_locally"]})

    if "user_prefs" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "user_prefs"),
            entry_point="main",
            parameters={
                "model": config["nn_arts"]["main_model"],
                "main_df": config["data"]["preprocessed_artifact_latest"],
                "project_name": config["main"]["project_name"],
                "anime_df": config["data"]["all_anime_artifact_latest"],
                "prefs_user_query": config["users"]["prefs_user_query"],
                "pref_random_user": config["users"]["pref_random_user"],
                "favorite_percentile": config["users"]["favorite_percentile"],
                "show_clouds": config["users"]["show_clouds"],
                "genre_fn": config["users"]["genre_fn"],
                "source_fn": config["users"]["source_fn"],
                "cloud_width": config["users"]["cloud_width"],
                "cloud_height": config["users"]["cloud_height"],
                "prefs_csv": config["users"]["prefs_csv"],
                "interval": config["main"]["interval"],
                "save_faves": config["users"]["save_faves"],
                "flow_user": config["users"]["flow_user"],
                "prefs_from_flow": config["users"]["prefs_from_flow"],
                "prefs_local_user": config["users"]["prefs_local_user"],
                "main_df_type": config["data"]["preprocessed_artifact_type"],
                "anime_df_type": config["data"]["all_anime_type"],
                "ID_type": config["users"]["ID_type"],
                "cloud_type": config["users"]["cloud_type"],
                "fave_art_type": config["users"]["fave_art_type"]})

    if "user_recs" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "user_recs"),
            entry_point="main",
            parameters={
                "main_df": config["data"]["preprocessed_artifact_latest"],
                "project_name": config["main"]["project_name"],
                "anime_df": config["data"]["all_anime_artifact_latest"],
                "user_recs_query": config["users"]["user_recs_query"],
                "user_recs_fn": config["users"]["user_recs_fn"],
                "save_user_recs": config["users"]["save_user_recs"],
                "sypnopses_df": config["data"]["sypnopses_artifact_latest"],
                "user_num_recs": config["users"]["user_num_recs"],
                "model": config["nn_arts"]["main_model"],
                "ID_emb_name": config["model"]["ID_emb_name"],
                "anime_emb_name": config["model"]["anime_emb_name"],
                "anime_df_type": config["data"]["all_anime_type"],
                "main_df_type": config["data"]["preprocessed_artifact_type"],
                "sypnopsis_df_type": config["data"]["synopses_artifact_type"],
                "model_type": config["nn_arts"]["model_type"],
                "user_recs_random": config["users"]["user_recs_random"],
                "recs_sim_from_flow": config["users"]["recs_sim_from_flow"],
                "user_recs_type": config["users"]["user_recs_type"],
                "recs_ID_from_flow": config["users"]["recs_ID_from_flow"],
                "flow_ID": config["users"]["flow_user"],
                "flow_ID_type": config["users"]["ID_type"],
                "sim_users_art": config["users"]["sim_users_latest"],
                "sim_users_art_type": config["users"]["sim_users_type"],
                "recs_n_sim_ID": config["users"]["recs_n_sim_ID"],
                "recs_ID_from_conf": config["users"]["recs_ID_from_conf"],
                "ID_rec_genres": config["users"]["ID_rec_genres"],
                "ID_spec_genres": config["users"]["ID_spec_genres"],
                "recs_pref_from_flow": config["users"]["recs_pref_from_flow"],
                "prefs_input_fn": config["users"]["prefs_input_fn"],
                "prefs_input_type": config["users"]["prefs_input_type"]})

    if "model_recs" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "model_recs"),
            entry_point="main",
            parameters={
                "main_df": config["data"]["preprocessed_artifact_latest"],
                "main_df_type": config["data"]["preprocessed_artifact_type"],
                "project_name": config["main"]["project_name"],
                "anime_df": config["data"]["all_anime_artifact_latest"],
                "anime_df_type": config["data"]['all_anime_type'],
                "sypnopsis_df": config["data"]["sypnopses_artifact_latest"],
                "sypnopsis_type": config["data"]["synopses_artifact_type"],
                "model": config["nn_arts"]["main_model"],
                "model_type": config["nn_arts"]["model_type"],
                "model_user_query": config["model_recs"]["model_user_query"],
                "random_user": config["model_recs"]["model_random_user"],
                "model_recs_fn": config["model_recs"]["model_recs_fn"],
                "save_model_recs": config["model_recs"]["save_model_recs"],
                "model_num_recs": config["model_recs"]["model_num_recs"],
                "anime_types": config["model_recs"]["anime_types"],
                "specify_types": config["model_recs"]['specify_types'],
                "model_genres": config["model_recs"]["model_genres"],
                "specify_genres": config["model_recs"]["specify_genres"],
                "model_ID_flow": config["model_recs"]["model_ID_flow"],
                "model_ID_conf": config["model_recs"]["model_ID_conf"],
                "model_recs_type": config["model_recs"]["model_recs_type"],
                "sypnopsis_df_type": config["data"]["synopses_artifact_type"],
                "flow_ID": config["users"]["flow_user"],
                "flow_ID_type": config["users"]["ID_type"]})


if __name__ == "__main__":
    go()
