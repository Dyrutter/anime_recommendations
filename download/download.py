name: download
conda_env: conda.yml

entry_points:
  main:
    parameters:
      stats_url:
        description: URL of the stats data file to download
        type: uri
      stats_artifact:
        description: Name for the W&B stats data artifact that will be created
        type: str
      stats_artifact_type:
        description: Type of the stats data artifact to create
        type: str
        default: raw_data
      stats_artifact_description:
        description: Description for the stats data artifact
        type: str
      synopses_url:
        description: URL of the synopses data file to download
        type: uri
      synopses_artifact:
        description: Name for the W&B synopses data artifact that will be created
        type: str
      synopses_artifact_type:
        description: Type of the synopses data artifact to create
        type: str
        default: raw_data
      synopses_artifact_description:
        description: Description for the synopses data artifact
        type: str
      all_anime_url:
        description: URL of the all_anime data file to download
        type: uri
      all_anime_artifact:
        description: Name for the W&B all_anime data artifact that will be created
        type: str
      all_anime_type:
        description: Type of the all_anime data artifact to create
        type: str
        default: raw_data
      all_anime_description:
        description: Description for the all_anime data artifact
        type: str
      save_raw_locally:
        description: Boolean of whether or not to save raw data to local machine
        type: str
      project_name:
        description: Name of wandb project
        type: str

    command: >-
      python download.py --stats_url {stats_url} \
                              --stats_artifact {stats_artifact} \
                              --stats_artifact_type {stats_artifact_type} \
                              --stats_artifact_description {stats_artifact_description} \
                              --synopses_url {synopses_url} \
                              --synopses_artifact {synopses_artifact} \
                              --synopses_artifact_type {synopses_artifact_type} \
                              --synopses_artifact_description {synopses_artifact_description} \
                              --all_anime_url {all_anime_url} \
                              --save_raw_locally {save_raw_locally} \
                              --all_anime_artifact {all_anime_artifact} \
                              --all_anime_type {all_anime_type} \
                              --project_name {project_name} \
                              --all_anime_description {all_anime_description}
