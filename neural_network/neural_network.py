import argparse
import logging
import wandb
from distutils.util import strtobool
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.layers as tfkl
import tensorflow.keras.callbacks as tfkc
import matplotlib.pyplot as plt


logging.basicConfig(
    filename='./neural_network.log',  # Path to log file
    level=logging.INFO,  # Log info, warnings, errors, and critical errors
    filemode='a',  # Create log file if one doesn't already exist and add
    format='%(asctime)s-%(name)s - %(levelname)s - %(message)s',
    datefmt='%d %b %Y %H:%M:%S %Z',  # Format date
    force=True)
logger = logging.getLogger()


def get_df():
    """
    Get data frame from wandb
    """
    run = wandb.init(project=args.project_name)
    logger.info("Downloading data artifact")
    artifact = run.use_artifact(args.input_data, type='preprocessed_data')
    artifact_path = artifact.file()
    df = pd.read_parquet(artifact_path)

    # Encoding categorical data
    # Get all user ids and anime ids
    user_ids = df["user_id"].unique().tolist()
    anime_ids = df["anime_id"].unique().tolist()

    # Dict of format {user_id: count_number}
    reverse_encoded_user_ids = {x: i for i, x in enumerate(user_ids)}
    reverse_encoded_anime = {x: i for i, x in enumerate(anime_ids)}

    # Convert values of format id to count_number
    df["user"] = df["user_id"].map(reverse_encoded_user_ids)
    df["anime"] = df["anime_id"].map(reverse_encoded_anime)

    # Get number of users and number of anime
    n_users = len(reverse_encoded_user_ids)
    n_animes = len(reverse_encoded_anime)

    # Shuffle, because currently sorted according to user ID
    df = df.sample(frac=1, random_state=42)
    df = df[['user', 'anime', 'rating']]
    logger.info("Final df shape is %s", df.shape)

    return df, n_users, n_animes


def neural_network():
    """
    Create a neural network embedding model. The purpose is to get
    embedding weights, the representation of users and anime as
    continuous vectors.
    """
    df, n_users, n_animes = get_df()
    # Both inputs are 1-dimensional
    user = tfkl.Input(name='user', shape=[1])
    user_embedding = tfkl.Embedding(name='user_embedding',
                                    input_dim=n_users,
                                    output_dim=int(args.embedding_size))(user)

    anime = tfkl.Input(name='anime', shape=[1])
    anime_embedding = tfkl.Embedding(
        name='anime_embedding',
        input_dim=n_animes,
        output_dim=int(args.embedding_size))(anime)

    # Merge layer with a dot product along second axis
    merged = tfkl.Dot(
        name='dot_product',
        normalize=True,
        axes=2)([user_embedding, anime_embedding])

    # Reshape to be a single number (shape will be (None, 1))
    merged = tfkl.Flatten()(merged)
    out = tfkl.Dense(
        1,
        kernel_initializer=args.kernel_initializer)(merged)

    norm = tfkl.BatchNormalization()(out)
    model = tfkl.Activation(args.activation_function)(norm)
    model = Model(inputs=[user, anime], outputs=model)
    model.compile(loss=args.model_loss,
                  metrics=["mae", "mse"],
                  optimizer=args.optimizer)
    logger.info("Model compiled!")
    return model


def lrfn(epoch):
    """
    Scheduel the learning rate for each epoch
    """
    max_lr = float(args.max_lr)
    if epoch < int(args.rampup_epochs):
        return (float(max_lr) - float(args.start_lr)) / \
            int(args.rampup_epochs) * epoch + float(args.start_lr)
    elif epoch < int(args.rampup_epochs) + int(args.sustain_epochs):
        return float(max_lr)
    else:
        return (float(max_lr) - float(args.min_lr)) * float(args.exp_decay) **\
            (epoch - int(args.rampup_epochs) - int(args.sustain_epochs))\
            + float(args.min_lr)


def extract_weights(name, model):
    """
    Extract weights from model
    """
    weight_layer = model.get_layer(name)
    weights = weight_layer.get_weights()[0]
    weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
    return weights


def go(args):
    if args.TPU_INIT is True:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
        logger.info('TPU initialized')
    else:
        logger.info("TPU is not being used")

    run = wandb.init(
        job_type="neural_network",
        project=args.project_name,
        name="400_nn_run")
    rating_df, n_users, n_anime = get_df()
    logger.info("Data frame loaded")

    # Get np array of user and anime columns (features) and ratings (labels)
    X = rating_df[['user', 'anime']].values
    y = rating_df["rating"]

    # Split
    rating_df = rating_df.sample(frac=1, random_state=73)
    train_indices = rating_df.shape[0] - int(args.test_size)
    X_train, X_test, y_train, y_test = (
        X[:train_indices],
        X[train_indices:],
        y[:train_indices],
        y[train_indices:])
    # Get arrays of anime IDs and User IDs
    X_train_array = [X_train[:, 0], X_train[:, 1]]
    X_test_array = [X_test[:, 0], X_test[:, 1]]

    max_lr = float(args.max_lr)
    batch_size = int(args.batch_size)
    if args.TPU_INIT is True:
        with tpu_strategy.scope():
            model = neural_network()
        max_lr = float(max_lr) * tpu_strategy.num_replicas_in_sync
        batch_size = batch_size * tpu_strategy.num_replicas_in_sync
        checkpoint_options = tf.train.CheckpointOptions(enable_async=True)

    else:
        model = neural_network()
        checkpoint_options = tf.train.CheckpointOptions(enable_async=False)

    lr_callback = tfkc.LearningRateScheduler(
        lambda epoch: lrfn(epoch),
        verbose=0)

    model_checkpoints = tfkc.ModelCheckpoint(
        filepath=args.weights_artifact,
        save_weights_only=args.save_weights_only,
        monitor=args.checkpoint_metric,
        save_freq=args.save_freq,
        mode=args.mode,
        save_best_only=args.save_best_weights,
        verbose=int(args.verbose),
        options=checkpoint_options)

    early_stopping = tfkc.EarlyStopping(patience=3,
                                        monitor=args.checkpoint_metric,
                                        mode=args.mode,
                                        restore_best_weights=True)

    model_callbacks = [
        model_checkpoints,
        lr_callback,
        early_stopping,
    ]

    # Model training
    history = model.fit(
        x=X_train_array,
        y=y_train,
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        verbose=int(args.verbose),
        validation_data=(X_test_array, y_test),
        callbacks=model_callbacks)

    # Save entire model to local machine, weights only otherwise
    if args.save_model is True:
        model.save(args.model_name)
    logger.info('model trained and saved!')
    hist_df = pd.DataFrame(history.history)

    # Save history to json:
    hist_json_file = 'history.json'
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    # Save history to csv:
    hist_csv_file = args.history_csv
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    # Save anime weights and user weights
    # anime_weights = extract_weights('anime_embedding', model)
    # user_weights = extract_weights('user_embedding', model)
    # anime_weights.tofile('./wandb_anime_weights.csv', sep=',')
    # user_weights.tofile('./wandb_user_weights.csv', sep=',')

    # Log all weights
    logger.info("creating all weights artifact")
    wandb_weights_artifact = wandb.Artifact(
        name=args.weights_artifact,
        type="h5",
        description='file containing all weights')
    wandb_weights_artifact.add_file(args.weights_artifact)
    run.log_artifact(wandb_weights_artifact)
    logger.info("Weights logged!")
    wandb_weights_artifact.wait()

    # Log history csv
    logger.info("creating history artifact")
    hist_artifact = wandb.Artifact(
        name=args.history_csv,
        type='history_csv',
        description='csv file of neural network training history')
    hist_artifact.add_file(args.history_csv)
    run.log_artifact(hist_artifact)
    logger.info('History logged!')
    hist_artifact.wait()

    # log model
    logger.info("Creating model artifact")
    wandb_model_artifact = wandb.Artifact(
        name=args.model_artifact,
        type='h5',
        description='h5 file of trained neural network')
    wandb_model_artifact.add_file(args.model_artifact)
    run.log_artifact(wandb_model_artifact)
    logger.info("Model logged!")
    wandb_model_artifact.wait()
    """

    # Log anime weights
    logger.info("creating anime weights artifact")
    wandb_anime_weights_artifact = wandb.Artifact(
        name='wandb_anime_weights.csv',
        type='numpy_array',
        description='numpy array of anime weights')
    wandb_anime_weights_artifact.add_file('wandb_anime_weights.csv')
    logger.info('Logging anime weights array')
    run.log_artifact(wandb_anime_weights_artifact)
    wandb_anime_weights_artifact.wait()

    # Log user weights
    logger.info("creating user weights artifact")
    wandb_user_weights_artifact = wandb.Artifact(
        name='wandb_user_weights.csv',
        type='numpy_array',
        description='numpy array of user id weights')
    wandb_user_weights_artifact.add_file('wandb_user_weights.csv')
    logger.info("Logging id weights array")
    run.log_artifact(wandb_user_weights_artifact)
    wandb_user_weights_artifact.wait()
    """

    plt.plot(history.history["loss"][0:-2])
    plt.plot(history.history["val_loss"][0:-2])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    fig = plt
    run.log(
        {
            "results": wandb.Image(fig),
        }
    )
    logger.info("Figure logged!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an anime recommendation neural network",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--test_size",
        type=str,
        help="Size of validation set in train_test_split",
        required=True
    )

    parser.add_argument(
        "--TPU_INIT",
        type=lambda x: bool(strtobool(x)),
        help="Boolean of whether or not to use TPU for training",
        required=True
    )

    parser.add_argument(
        "--embedding_size",
        type=str,
        help="Size of embedding to use in neural network",
        required=True
    )

    parser.add_argument(
        "--kernel_initializer",
        type=str,
        help="kernal initializer to use for neural network",
        required=True
    )

    parser.add_argument(
        "--activation_function",
        type=str,
        help="activation function to use in neural network",
        required=True
    )

    parser.add_argument(
        "--model_loss",
        type=str,
        help="loss metric to use in neural network",
        required=True
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        help="Optimizer to use in neural network",
        required=True
    )

    parser.add_argument(
        "--start_lr",
        type=str,
        help="Initial learning rate to use",
        required=True
    )

    parser.add_argument(
        "--min_lr",
        type=str,
        help="Minimum learning rate to use",
        required=True
    )

    parser.add_argument(
        "--max_lr",
        type=str,
        help="Maximum learning rate to use",
        required=True
    )

    parser.add_argument(
        "--batch_size",
        type=str,
        help="Size of batches to use in neural network",
        required=True
    )

    parser.add_argument(
        "--rampup_epochs",
        type=str,
        help="Number of rampup epochs to use",
        required=True
    )

    parser.add_argument(
        "--sustain_epochs",
        type=str,
        help="Number of sustain epochs to use",
        required=True
    )

    parser.add_argument(
        "--exp_decay",
        type=str,
        help="Exponential decay rate to use",
        required=True
    )

    parser.add_argument(
        "--weights_artifact",
        type=str,
        help="Name of checkpoint artifact to use to save weights",
        required=True
    )

    parser.add_argument(
        "--save_weights_only",
        type=lambda x: bool(strtobool(x)),
        help="Whether or not to save weights only in checkpoint",
        required=True
    )

    parser.add_argument(
        "--checkpoint_metric",
        type=str,
        help="Metric to monitor at each checkpoint",
        required=True
    )

    parser.add_argument(
        "--save_freq",
        type=str,
        help="Frequency with which to monitor and save",
        required=True
    )

    parser.add_argument(
        "--mode",
        type=str,
        help="How to evaluate the metric of an epoch, e.g. min or max",
        required=True
    )

    parser.add_argument(
        "--save_best_weights",
        type=lambda x: bool(strtobool(x)),
        help="Boolean save best weights if True, all weights if False",
        required=True
    )

    parser.add_argument(
        "--verbose",
        type=str,
        help="Print progress stats (1) or don't (0)",
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=str,
        help="Number of epochs to run",
        required=True
    )

    parser.add_argument(
        "--save_model",
        type=lambda x: bool(strtobool(x)),
        help="Boolean of whether or not to save model",
        required=True
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="Path and name to which to save model",
        required=True
    )

    parser.add_argument(
        "--input_data",
        type=str,
        help="Preprocessed data artifact with which to build model",
        required=True
    )

    parser.add_argument(
        "--project_name",
        type=str,
        help="Name of wandb project",
        required=True
    )

    parser.add_argument(
        "--model_artifact",
        type=str,
        help="name of model artifact to save in wandb",
        required=True
    )

    parser.add_argument(
        "--history_csv",
        type=str,
        help="Name of model history csv file to save",
        required=True
    )

    args = parser.parse_args()
    go(args)
