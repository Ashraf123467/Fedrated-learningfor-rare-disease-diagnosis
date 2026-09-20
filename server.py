"""
server.py - Central Federated Server & Federated Averaging (FedAvg) Orchestrator.

Implements FedAvg (McMahan et al.):
  W_global = sum( (n_k / N) * W_k )
  b_global = sum( (n_k / N) * b_k )

Where n_k is number of training samples at Hospital k, and N is total samples across all clients.
Flower FL integration ready structure.
"""

import numpy as np
from model import RareDiseaseClassifier
from data import FEATURE_NAMES

class FederatedServer:
    """
    Central Coordinator for Federated Communication Rounds.
    """
    def __init__(self, clients: list, X_global_test, y_global_test, random_state: int = 42):
        self.clients = clients
        self.X_global_test = X_global_test
        self.y_global_test = y_global_test
        self.global_model = RareDiseaseClassifier(random_state=random_state)
        
        # Initialize global model parameters (Zero initialized)
        n_features = len(FEATURE_NAMES)
        self.global_parameters = {
            "coef": np.zeros((1, n_features)),
            "intercept": np.zeros((1,))
        }
        self.history = []

    def federated_averaging(self, client_updates: list) -> dict:
        """
        Performs Federated Averaging (FedAvg) over client parameter updates.
        
        client_updates: List of tuples [(params_k, n_k), ...]
        """
        total_samples = sum(n_k for _, n_k in client_updates)
        
        # Initialize accumulators
        aggregated_coef = np.zeros_like(self.global_parameters["coef"])
        aggregated_intercept = np.zeros_like(self.global_parameters["intercept"])

        for params_k, n_k in client_updates:
            weight = n_k / total_samples
            aggregated_coef += weight * params_k["coef"]
            aggregated_intercept += weight * params_k["intercept"]

        return {
            "coef": aggregated_coef,
            "intercept": aggregated_intercept
        }

    def run_communication_round(self, round_num: int) -> dict:
        """
        Executes a single Federated Learning Communication Round:
        1. Server broadcasts `global_parameters` to all clients.
        2. Clients perform local training on private data.
        3. Clients send back local parameter weights & sample count.
        4. Server executes FedAvg parameter aggregation.
        5. Server updates global model and evaluates global performance metrics.
        """
        client_updates = []
        client_local_metrics = {}

        # Broadcast & Local Training
        for client in self.clients:
            updated_params, n_k = client.train_local(self.global_parameters)
            client_updates.append((updated_params, n_k))
            client_local_metrics[client.hospital_id] = client.evaluate_local()

        # Aggregation via FedAvg
        self.global_parameters = self.federated_averaging(client_updates)
        
        # Update Central Global Model
        self.global_model.set_parameters(self.global_parameters)
        
        # Evaluate Global Model on Pooled Benchmark Test Set
        global_eval = self.global_model.evaluate(self.X_global_test, self.y_global_test)

        round_log = {
            "round": round_num,
            "global_metrics": global_eval,
            "client_metrics": client_local_metrics,
            "parameters": {
                "coef": self.global_parameters["coef"].copy(),
                "intercept": self.global_parameters["intercept"].copy()
            }
        }
        self.history.append(round_log)
        return round_log

    def run_federated_learning(self, num_rounds: int = 5) -> list:
        """
        Runs multi-round Federated Learning simulation.
        """
        self.history = []
        for r in range(1, num_rounds + 1):
            self.run_communication_round(round_num=r)
        return self.history