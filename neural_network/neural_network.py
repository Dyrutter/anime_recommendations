import argparse
import logging
# import os
# import wandb
from distutils.util import strtobool
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import tensorflow.keras.layers as tfkl
import tensorflow.keras.callbacks as tfkc
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()
if args.TPU_INIT is True:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    logger.info('TPU initialized')
else:
    logger.info("TPU is not being used")

df = pd.read_parquet(
    '/Applications/python_files/anime_recommendations/data/preprocessed_stats_full.parquet')

# Encoding categorical data
# Get all user ids and anime ids
user_ids = df["user_id"].unique().tolist()
anime_ids = df["anime_id"].unique().tolist()

# Dict of format {user_id: count_number}
reverse_encoded_user_ids = {x: i for i, x in enumerate(user_ids)}
reverse_encoded_anime = {x: i for i, x in enumerate(anime_ids)}

# Dict of format  {count_number: user_id}
encoded_user_ids = {i: x for i, x in enumerate(user_ids)}
encoded_anime = {i: x for i, x in enumerate(anime_ids)}

# Convert values of format id to count_number
df["user"] = df["user_id"].map(reverse_encoded_user_ids)
df["anime"] = df["anime_id"].map(reverse_encoded_anime)
n_users = len(reverse_encoded_user_ids)
n_animes = len(reverse_encoded_anime)

# Get min and max ratings
logger.info(f'Num of users: {n_users}, Num of animes: {n_animes}')
logger.info(
    f"Min rating: {min(df['rating'])}, Max rating: {max(df['rating'])}")

# Shuffle, because currently sorted according to user_id
df = df.sample(frac=1, random_state=42)
rating_df = df.drop(
    ['watching_status', 'watched_episodes', 'max_eps', 'half_eps'], axis=1)

# Holdout test set is approximately 10% of data set
test_df = rating_df[38000000:]
rating_df = rating_df[:38000000]

# Get np array of user and anime columns (features) and ratings (labels)
X = rating_df[['user', 'anime']].values
y = rating_df["rating"]

# Split into train and validation sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=int(args.test_size),
    random_state=42)

# Get arrays of anime IDs and User IDs
X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]


def neural_network():

    user = tfkl.Input(name='user', shape=[1])
    user_embedding = tfkl.Embedding(name='user_embedding',
                                    input_dim=n_users,
                                    output_dim=args.embedding_size)(user)

    anime = tfkl.Input(name='anime', shape=[1])
    anime_embedding = tfkl.Embedding(
        name='anime_embedding',
        input_dim=n_animes,
        output_dim=int(args.embedding_size))(anime)

    # x = tfkl.Concatenate()([user_embedding, anime_embedding])
    x = tfkl.Dot(
        name='dot_product',
        normalize=True,
        axes=2)([user_embedding, anime_embedding])
    x = tfkl.Flatten()(x)

    x = tfkl.Dense(1, kernel_initializer=args.kernal_initializer)(x)

    x = tfkl.BatchNormalization()(x)
    x = tfkl.Activation(args.activation_function)(x)

    model = Model(inputs=[user, anime], outputs=x)
    model.compile(loss=args.model_loss,
                  metrics=["mae", "mse"],
                  optimizer=args.optimizer)

    return model

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

max_lr = float(args.max_lr)
batch_size = int(args.batch_size)


def lrfn(epoch):
    if epoch < int(args.rampup_epochs):
        return (float(max_lr) - float(args.start_lr)) / \
            int(args.rampup_epochs) * epoch + float(args.start_lr)
    elif epoch < int(args.rampup_epochs) + int(args.sustain_epochs):
        return float(max_lr)
    else:
        return (float(max_lr) - float(args.min_lr)) * float(args.exp_decay) **\
            (epoch - int(args.rampup_epochs) - int(args.sustain_epochs))\
            + float(args.min_lr)


lr_callback = tfkc.LearningRateScheduler(
    lambda epoch: lrfn(epoch),
    verbose=int(args.verbose))
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
                                    monitor=args.monitor,
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

hist_df = pd.DataFrame(history.history)
# Save history to json:
hist_json_file = 'history.json'
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# Save history to csv:
hist_csv_file = args.history_csv
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


# model = tf.keras.models.load_model('./anime_nn.h5')
# weight = model.load_weights('./anime_weights.h5') #args.checkpoint_artifact
# weight_layer = model.get_layer('anime_embedding')
# weights = weight_layer.get_weights()[0]
# weights = weights / np.linalg.norm(weights, axis = 1).reshape((-1, 1))
# loss, acc = model.evaluate(X_test_array, y_test, verbose=2)
plt.plot(history.history["loss"][0:-2])
plt.plot(history.history["val_loss"][0:-2])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

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
        tpye=str,
        help="activation function to use in neural network",
        required=True
    )

    parser.add_argument(
        "--model_loss",
        type=str,
        description="loss metric to use in neural network",
        required=True
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        description="Optimizer to use in neural network",
        required=True
    )

    parser.add_argument(
        "--start_lr",
        type=str,
        description="Initial learning rate to use",
        required=True
    )

    parser.add_argument(
        "--min_lr",
        type=str,
        description="Minimum learning rate to use",
        required=True
    )

    parser.add_argument(
        "--max_lr",
        type=str,
        description="Maximum learning rate to use",
        required=True
    )

    parser.add_argument(
        "--batch_size",
        type=str,
        description="Size of batches to use in neural network",
        required=True
    )

    parser.add_argument(
        "--rampup_epochs",
        type=str,
        description="Number of rampup epochs to use",
        required=True
    )

    parser.add_argument(
        "--sustain_epochs",
        type=str,
        description="Number of sustain epochs to use",
        required=True
    )

    parser.add_argument(
        "--exp_decay",
        type=str,
        description="Exponential decay rate to use",
        required=True
    )

    parser.add_argument(
        "--weights_artifact",
        type=str,
        description="Name of checkpoint artifact to use to save weights",
        required=True
    )

    parser.add_argument(
        "--save_weights_only",
        type=lambda x: bool(strtobool(x)),
        description="Whether or not to save weights only in checkpoint",
        required=True
    )

    parser.add_argument(
        "--checkpoint_metric",
        type=str,
        description="Metric to monitor at each checkpoint",
        required=True
    )

    parser.add_argument(
        "--save_freq",
        type=str,
        description="Frequency with which to monitor and save",
        required=True
    )

    parser.add_argument(
        "--mode",
        type=str,
        description="How to evaluate the metric of an epoch, e.g. min or max",
        required=True
    )

    parser.add_argument(
        "--save_best_weights",
        type=lambda x: bool(strtobool(x)),
        description="Boolean save best weights if True, all weights if False",
        required=True
    )

    parser.add_argument(
        "--verbose",
        type=str,
        description="Print progress stats (1) or don't (0)",
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=str,
        description="Number of epochs to run",
        required=True
    )

    parser.add_argument(
        "--save_model",
        type=lambda x: bool(strtobool(x)),
        description="Boolean of whether or not to save model",
        required=True
    )

    parser.add_argument(
        "--model_name",
        type=str,
        description="Path and name to which to save model",
        required=True
    )

    parser.add_argument(
        "--input_data",
        type=str,
        description="Preprocessed data artifact with which to build model",
        required=True
    )

    parser.add_argument(
        "--project_name",
        type=str,
        description="Name of wandb project",
        required=True
    )

    parser.add_argument(
        "--model_artifact",
        type=str,
        description="name of model artifact to save in wandb",
        required=True
    )

    parser.add_argument(
        "--history_csv",
        type=str,
        description="Name of model history csv file to save",
        required=True
    )

if __name__=='__main__':
    args = parser.parse_args()
