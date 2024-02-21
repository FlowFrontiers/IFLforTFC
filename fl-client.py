import flwr as fl
import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import argparse
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
import numpy as np
import json
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers.legacy import Adam

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) `tensorflow` random seed
# 3) `python` random seed
keras.utils.set_random_seed(1)

# This will make TensorFlow ops as deterministic as possible, but it will
# affect the overall performance, so it's not enabled by default.
# `enable_op_determinism()` is introduced in TensorFlow 2.9.
tf.config.experimental.enable_op_determinism()

# Define FNN Model
# def create_model(input_dim: int, num_classes: int, learning_rate: float) -> keras.Model:
#     model = keras.Sequential([
#         keras.layers.Dense(32, activation='relu', input_dim=input_dim),
#         keras.layers.Dense(64, activation='relu'),
#         keras.layers.Dense(num_classes, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
#     return model

def create_model(input_dim: int, num_classes: int, learning_rate: float) -> keras.Model:
    hidden_layer1_neurons = int((2 / 3) * input_dim + num_classes)  # 2/3 of the input layer + size of the output layer
    hidden_layer2_neurons = input_dim  # matches the size of the Input Layer

    model = Sequential([
        InputLayer(input_shape=(input_dim,)),  # Defining Input Layer
        Dense(hidden_layer1_neurons, activation='relu'),  # Hidden Layer 1
        Dense(hidden_layer2_neurons, activation='relu'),  # Hidden Layer 2
        Dense(num_classes, activation='softmax')  # Output Layer
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model

class MetricsCallback(Callback):
    def __init__(self, validation_data, val_metrics):
        super().__init__()
        self.x_val, self.y_val = validation_data  # Keeping as self.y_val as per preference
        self.val_metrics = val_metrics

    def on_epoch_end(self, epoch, logs=None):
        # Get model predictions for the validation set
        y_pred_probs = self.model.predict(self.x_val)
        # Convert predicted probabilities to class labels
        y_pred = np.argmax(y_pred_probs, axis=-1)

        # Compute metrics (no need to convert self.y_val)
        precision = precision_score(self.y_val, y_pred, average='macro', zero_division=0)
        recall = recall_score(self.y_val, y_pred, average='macro', zero_division=0)
        f1 = f1_score(self.y_val, y_pred, average='macro', zero_division=0)

        # Extract validation loss from logs
        val_loss = logs.get('val_loss', np.nan)
        val_accuracy = logs.get('val_sparse_categorical_accuracy', np.nan)

        # Compute normalized confusion matrix
        normalized_cm = confusion_matrix(self.y_val, y_pred, normalize='true')

        # Store computed metrics and validation loss
        self.val_metrics["loss"].append(val_loss)
        self.val_metrics["accuracy"].append(val_accuracy)
        self.val_metrics["precision"].append(precision)
        self.val_metrics["recall"].append(recall)
        self.val_metrics["f1_score"].append(f1)
        self.val_metrics["confusion_matrix"].append(normalized_cm.tolist())


class MyClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, scenario: str, model) -> None:
        self.client_id = client_id
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

    def get_parameters(self, config):
        print("Setting model weights")
        return self.model.get_weights()

    def load_data_chunk(self, round_num: int, is_validation: bool = False):
        folder_name = f'datasets/{scenario}/validation_data_chunks' if is_validation else f"datasets/{scenario}/{self.client_id}_data_chunks"
        file_name = f"{scenario}_chunk_{round_num}.csv"
        path = os.path.join(folder_name, file_name)

        print(f"Loading data chunk {path}")

        if os.path.exists(path):
            df = pd.read_csv(path)
            X = df.drop('application_name', axis=1).values
            y = df['application_name'].values

            le = LabelEncoder()
            y = le.fit_transform(y)

            sc = StandardScaler()
            X = sc.fit_transform(X)

            return X, y

        else:
            print(f"Data chunk {path} does not exist")
            return None, None

    def fit(self, parameters, config):
        current_round = config['current_round']
        print(f"Current Round: {current_round}")
        print(f"Training the model locally")

        self.x_train, self.y_train = self.load_data_chunk(current_round)
        self.x_val, self.y_val = self.load_data_chunk(current_round, is_validation=True)

        if self.x_train is None:
            return parameters, 0, {}

        self.model.set_weights(parameters)

        # # Print received model weights
        # weights = model.get_weights()
        # for i, layer_weights in enumerate(weights):
        #     print(f"Layer {i} weights:\n", layer_weights)

        # Initialize dictionaries to store metrics
        train_metrics = {'loss': [], 'accuracy': []}
        val_metrics = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
                       'confusion_matrix': []}


        # Initialize the callback with validation data and metric storage
        metrics_callback = MetricsCallback(validation_data=(self.x_val, self.y_val), val_metrics=val_metrics)

        history = self.model.fit(self.x_train, self.y_train,
                                 epochs=10, batch_size=32, verbose=2,
                                 validation_data=(self.x_val, self.y_val),
                                 callbacks=[metrics_callback])

        # Append train_loss and train_accuracy to the metric_storage from history
        train_metrics['loss'].extend(history.history['loss'])
        train_metrics['accuracy'].extend(history.history['sparse_categorical_accuracy'])

        # After all epochs are done, save the metrics to file
        filename = f"results/{scenario}_fl_{self.client_id}_results.json"

        try:
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {
                "train_metrics": {'loss': [], 'accuracy': []},
                "val_metrics": {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
                                'confusion_matrix': []}
            }

        # Combine existing data with the new one
        for key in train_metrics.keys():
            existing_data['train_metrics'][key].extend(train_metrics[key])

        for key in val_metrics.keys():
            existing_data['val_metrics'][key].extend(val_metrics[key])

        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=4)

        loss = history.history['loss'][-1]
        accuracy = history.history['sparse_categorical_accuracy'][-1]

        return self.model.get_weights(), len(self.x_train), {"loss": loss, "accuracy": accuracy}

    def evaluate(self, parameters, config):
        print(f"Validating the global model for the server")
        current_round = config['current_round']  # Access the current round from config
        self.x_val, self.y_val = self.load_data_chunk(current_round, is_validation=True)

        if self.x_val is None:
            return 0.0, 0, {}

        self.model.set_weights(parameters)

        # Get model predictions
        y_pred = self.model.predict(self.x_val)
        y_pred_classes = y_pred.argmax(axis=1)

        loss, accuracy = self.model.evaluate(self.x_val, self.y_val, verbose=2)

        # Calculate accuracy, precision, recall, and F1 score
        precision = precision_score(self.y_val, y_pred_classes, average='macro', zero_division=0)
        recall = recall_score(self.y_val, y_pred_classes, average='macro', zero_division=0)
        f1 = f1_score(self.y_val, y_pred_classes, average='macro', zero_division=0)
        normalized_cm = confusion_matrix(self.y_val, y_pred_classes, normalize='true').tolist()

        val_metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": normalized_cm,
        }

        # File to save metrics
        filename = f"results/{scenario}_fl_{self.client_id}_validates_for_server.json"

        try:
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {
                "val_metrics": {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
                                'confusion_matrix': []}
            }

        # Combine existing data with the new one
        for key in val_metrics.keys():
            existing_data['val_metrics'][key].append(val_metrics[key])

        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=4)

        return loss, len(self.x_val), {"loss": loss, "accuracy": accuracy}


if __name__ == "__main__":
    model = create_model(input_dim=14, num_classes=10, learning_rate=0.001)
    parser = argparse.ArgumentParser(description='Start a Flower client in a scenario with a unique ID.')
    parser.add_argument('--client_id', type=str, help='Unique ID for the client.')
    parser.add_argument('--scenario', type=str, help='The scenario in which the client is run.')

    args = parser.parse_args()
    client_id = args.client_id
    scenario = args.scenario
    print(client_id)

    client = MyClient(client_id=client_id, scenario=scenario, model=model)
    fl.client.start_numpy_client(server_address="0.0.0.0:8687", client=client)
