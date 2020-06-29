import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import argparse
import os
import numpy as np
import json


def model(X_train, y_train, X_val, y_val):
    model = models.Sequential()
    # Conv 32x32x1 => 28x28x6.
    model.add(layers.Conv2D(filters = 6, kernel_size = (5, 5), strides=(1, 1), padding='valid',
                            activation='relu', data_format = 'channels_last', input_shape = (32, 32, 1)))
    # Maxpool 28x28x6 => 14x14x6
    model.add(layers.MaxPooling2D((2, 2)))
    # Conv 14x14x6 => 10x10x16
    model.add(layers.Conv2D(16, (5, 5), activation='relu'))
    # Maxpool 10x10x16 => 5x5x16
    model.add(layers.MaxPooling2D((2, 2)))
    # Flatten 5x5x16 => 400
    model.add(layers.Flatten())
    # Fully connected 400 => 120
    model.add(layers.Dense(120, activation='relu'))
    # Fully connected 120 => 84
    model.add(layers.Dense(84, activation='relu'))
    # Dropout
    model.add(layers.Dropout(0.2))
    # Fully connected, output layer 84 => 43
    model.add(layers.Dense(43, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=128, epochs=10,
                    validation_data=(X_val, y_val))

    return model


def _load_training_data(base_dir):
    X_train = np.load(os.path.join(base_dir, 'training.npy'))
    y_train = np.load(os.path.join(base_dir, 'training_label.npy'))
    return X_train, y_train


def _load_validation_data(base_dir):
    X_val = np.load(os.path.join(base_dir, 'validation.npy'))
    y_val = np.load(os.path.join(base_dir, 'validation_label.npy'))
    return X_val, y_val


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    train_data, train_labels = _load_training_data(args.train)
    eval_data, eval_labels = _load_validation_data(args.train)

    mdl = model(train_data, train_labels, eval_data, eval_labels)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        mdl.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')
