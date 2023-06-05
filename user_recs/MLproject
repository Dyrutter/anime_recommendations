name: user_recs
conda_env: conda.yml

entry_points:
  main:
    parameters:
      model:
        description: Neural network model artifact path
        type: str
      project_name:
        description: Name of Wandb project
        type: str
      main_df:
        description: Main preprocessed data frame artifact
        type: str
      anime_df:
        description: Name of anime data frame artifact
        type: str
      user_recs_query:
        description: User of which to find similar animes to
        type: str
      user_recs_fn:
        description: Artifact name of user-based recommendations csv file to create
        type: str
      save_user_recs:
        description: Whether or not to save user recommendations csv file to local machine
        type: str
      sypnopses_df:
        description: Artifact name of sypnopsis data frame
        type: str
      user_num_recs:
        description: Number of anime recommendations to return in user_recs.py
        type: str 
      ID_emb_name:
        description: Name of User ID embedding layer in neural network model
        type: str
      anime_emb_name:
        description: Name of anime ID embedding layer in neural network model
        type: str
      main_df_type:
        description: Artifact type of main preprocessed data frame
        type: str
      anime_df_type:
        description: Artifact type of anime data frame
        type: str
      sypnopsis_df_type:
        description: Artifact type of sypnopsis data frame
        type: str
      model_type:
        description: Type of model artifact
        type: str
      user_recs_random:
        description: Boolean of whether to use a random user for recommendations
        type: str
      recs_sim_from_flow:
        description: Boolean of whether to use the similar users artifact found in MLflow run
        type: str
      user_recs_type:
        description: Type of artifact to save anime recommendations file as
        type: str
      recs_ID_from_flow:
        description: Whether to use the User ID artifact created in MLflow
        type: str
      flow_ID:
        description: ID of MLflow artifact name to use if recs_ID_from_flow is True
        type: str
      flow_ID_type:
        description: Type of mlflow artifact user ID was saved as
        type: str
      sim_users_art:
        description: Name of similar users artifact to be used if recs_sim_from_flow is True
        type: str
      sim_users_art_type:
        description: type of similar users artifact to be used if recs_sim_from_flow is True
        type: str 
    command: >-
      python user_recs.py --project_name {project_name} \
                              --main_df {main_df} \
                              --anime_df {anime_df} \
                              --user_recs_query {user_recs_query} \
                              --user_recs_fn {user_recs_fn} \
                              --save_user_recs {save_user_recs} \
                              --sypnopses_df {sypnopses_df} \
                              --user_num_recs {user_num_recs} \
                              --model {model} \
                              --ID_emb_name {ID_emb_name} \
                              --anime_emb_name {anime_emb_name} \
                              --main_df_type {main_df_type} \
                              --anime_df_type {anime_df_type} \
                              --sypnopsis_df_type {sypnopsis_df_type} \
                              --model_type {model_type} \
                              --user_recs_random {user_recs_random} \
                              --recs_sim_from_flow {recs_sim_from_flow} \
                              --user_recs_type {user_recs_type} \
                              --recs_ID_from_flow {recs_ID_from_flow} \
                              --flow_ID {flow_ID} \
                              --flow_ID_type {flow_ID_type} \
                              --sim_users_art {sim_users_art} \
                              --sim_users_art_type {sim_users_art_type}
