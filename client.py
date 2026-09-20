"""
client.py - Hospital Client Node in Federated Learning System.

Represents an on-premise hospital client node (e.g. St. Jude, Mayo, Johns Hopkins).
Key Federated Learning Principle:
- Patient data remains isolated inside `HospitalClient` instance.
- Only model parameter dictionaries (`coef`, `intercept`) are sent across the boundary.
"""

from model import RareDiseaseClassifier
from data import HOSPITAL_METADATA

class HospitalClient:
    """
    On-premise Hospital Node simulating privacy-preserving local training.
    """
    def __init__(self, hospital_id: str, X_train, y_train, X_val, y_val, random_state: int = 42):
        self.hospital_id = hospital_id
        self.metadata = HOSPITAL_METADATA[hospital_id]
        self.hospital_name = self.metadata["name"]
        
        # PRIVATE LOCAL DATASET (Never transmitted over network)
        self._X_train = X_train
        self._y_train = y_train
        self._X_val = X_val
        self._y_val = y_val
        
        # Local ML Model Instance
        self.local_model = RareDiseaseClassifier(random_state=random_state)
        self.num_samples = len(X_train)

    def train_local(self, global_parameters: dict = None) -> tuple:
        """
        Federated Learning Step:
        1. Receive updated global model parameters from central server.
        2. Set local parameters to global state.
        3. Train model locally on private hospital dataset.
        4. Return updated parameter weights and sample count.
        """
        if global_parameters is not None:
            self.local_model.set_parameters(global_parameters)

        # Local On-Premise Training
        self.local_model.fit(self._X_train, self._y_train)
        
        # Extract weights to send back to server (Privacy preserving: No raw data transferred!)
        updated_params = self.local_model.get_parameters()
        return updated_params, self.num_samples

    def evaluate_local(self) -> dict:
        """
        Evaluates local model performance on local hospital validation set.
        """
        return self.local_model.evaluate(self._X_val, self._y_val)

    def get_data_summary(self) -> dict:
        """
        Returns metadata summary ONLY (no raw rows/features revealed).
        """
        return {
            "hospital_id": self.hospital_id,
            "hospital_name": self.hospital_name,
            "train_samples": len(self._X_train),
            "val_samples": len(self._X_val),
            "positive_cases": int(self._y_train.sum()),
            "negative_cases": int(len(self._y_train) - self._y_train.sum()),
            "raw_data_transmitted_bytes": 0  # Guarantees Privacy Boundary!
        }
    