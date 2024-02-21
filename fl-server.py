import flwr as fl
# from flwr.common import Scalar
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers.legacy import Adam
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import json
import argparse
import random

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# SEED = 42
#
# def set_seeds(seed: int = SEED):
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#     os.environ['TF_DETERMINISTIC_OPS'] = '1'  # This can help in achieving deterministic behavior
#
# set_seeds(SEED)

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

def on_fit_config_fn(rnd: int) -> Dict[str, fl.common.Scalar]:
    """Send the current round to the client."""
    # # Print model weights
    # weights = server.model.get_weights()
    # for i, layer_weights in enumerate(weights):
    #     print(f"Layer {i} weights:\n", layer_weights)
    return {'current_round': rnd}

# Define a function to aggregate fit metrics
def aggregate_fit_metrics(metrics):
    # Here, perform the aggregation of the received fit metrics from different clients.
    print(metrics)
    num_clients = len(metrics)
    total_loss = sum(metric[1]["loss"] for metric in metrics) / num_clients
    total_accuracy = sum(metric[1]["accuracy"] for metric in metrics) / num_clients
    return {"loss": total_loss, "accuracy": total_accuracy}

class FedAvgServer:
    def __init__(self, scenario):
        self.model = create_model(input_dim=14, num_classes=10, learning_rate=0.001)
        self.scenario = scenario

    def global_evaluation(self, model: keras.Model, round: int) -> Tuple[float, float]:
        # Construct the file name based on the round number
        filename = f"datasets/{self.scenario}/validation_data_chunks/{self.scenario}_chunk_{round}.csv"
        print(f"Loading data chunk {filename}")

        # Load the dataset
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} does not exist.")

        df = pd.read_csv(filename)
        X = df.drop('application_name', axis=1).values
        y = df['application_name'].values

        # # Check the unique classes in the target variable
        # unique_classes = np.unique(y)
        # num_classes = 10
        # # If the number of unique classes is less than the total number of classes,
        # # some classes are not represented
        # if len(unique_classes) < num_classes:
        #     missing_classes = set(range(num_classes)) - set(unique_classes)
        #     print(f"Missing classes: {missing_classes}")
        # else:
        #     print("All classes are represented.")

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Get model predictions for the validation set and
        # convert predicted probabilities to class labels
        y_pred = np.argmax(model.predict(X_scaled), axis=-1)

        # Evaluate the model to get loss and accuracy
        # loss, _ = model.evaluate(X_scaled, y_encoded, verbose=0)
        loss, accuracy = model.evaluate(X_scaled, y_encoded, verbose=0)

        # Compute metrics (no need to convert y_encoded)
        # accuracy = accuracy_score(y_encoded, y_pred)
        precision = precision_score(y_encoded, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_encoded, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_encoded, y_pred, average='macro', zero_division=0)

        # Compute normalized confusion matrix
        normalized_cm = confusion_matrix(y_encoded, y_pred, normalize='true').tolist()

        # Prepare metrics dictionary
        val_metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": normalized_cm,
        }

        return val_metrics

    # def eval_fn(server_round: int, weights: fl.common.Parameters, config: Dict) -> Optional[fl.common.EvaluateRes]:
    def eval_fn(self, server_round: int, weights: fl.common.Parameters, config: Dict) -> Optional[
            Tuple[float, Dict[str, fl.common.Scalar]]]:

        if server_round == 0:
            print("Skipping evaluation for server round 0")
            return None  # Skipping evaluation at round 0

        print("Server Round:", server_round)

        # Set the model weights
        self.model.set_weights(weights)

        # Perform the evaluation based on the server round
        val_metrics = self.global_evaluation(model=self.model, round=server_round)

        # Log metrics
        # print("Evaluation metrics:")
        # for key, value in val_metrics.items():
        #     if key != "confusion_matrix":
        #         print(f"{key}: {value}")

        # Store metrics to a JSON file similar to the client-side
        filename = f"results/{self.scenario}_fl_server_results.json"

        try:
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {'val_metrics': {'loss': [],
                                             'accuracy': [],
                                             'precision': [],
                                             'recall': [],
                                             'f1_score': [],
                                             'confusion_matrix': []
                                             }}

        # Combine existing data with the new one
        for key in val_metrics.keys():
            existing_data['val_metrics'][key].append(val_metrics[key])

        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=4)

        # Return loss and metrics dictionary
        return val_metrics["loss"], {"accuracy": val_metrics['accuracy']}
        # return metrics_dict["loss"], metrics_dict

    def run(self):
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=0.5,
            fraction_evaluate=0.0, # The fraction of clients that will be selected for evaluation. If fraction_evaluate is set to 0.0, federated evaluation will be disabled.
            min_fit_clients=5,
            min_available_clients=5,
            # on_fit_config_fn=lambda _: {},
            on_fit_config_fn=on_fit_config_fn,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,  # Use the defined aggregation function here
            on_evaluate_config_fn=on_fit_config_fn,  # add this line with the appropriate function to send the current round
            evaluate_metrics_aggregation_fn = aggregate_fit_metrics,  # You can use the same or a different function here
            evaluate_fn = self.eval_fn  # Pass the evaluation function here
        )

        fl.server.start_server(
            server_address="0.0.0.0:8687",
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=10)
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start a Flower server in a scenario.')
    parser.add_argument('--scenario', type=str, help='The scenario in which the server is run.')

    args = parser.parse_args()
    scenario = args.scenario

    server = FedAvgServer(scenario)
    server.run()
