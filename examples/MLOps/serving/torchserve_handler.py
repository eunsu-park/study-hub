"""
TorchServe Custom Handler Example
=================================

Custom handler example for TorchServe.

Usage:
    1. Create model archive:
       torch-model-archiver --model-name mymodel \\
           --version 1.0 \\
           --serialized-file model.pt \\
           --handler torchserve_handler.py \\
           --export-path model_store

    2. Start TorchServe:
       torchserve --start --model-store model_store --models mymodel=mymodel.mar

    3. Send prediction request:
       curl -X POST http://localhost:8080/predictions/mymodel \\
           -H "Content-Type: application/json" \\
           -d '{"data": [1.0, 2.0, 3.0, 4.0]}'
"""

import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler
import json
import logging
import os
import time

logger = logging.getLogger(__name__)


class ChurnPredictionHandler(BaseHandler):
    """
    Customer churn prediction model handler

    This handler performs the following:
    1. Model initialization and loading
    2. Input data preprocessing
    3. Inference execution
    4. Result postprocessing
    """

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.model = None
        self.device = None
        self.class_names = None
        self.feature_names = None

    def initialize(self, context):
        """
        Initialize model

        Args:
            context: TorchServe context object
        """
        logger.info("Initializing model...")

        # Extract info from context
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Device setup
        if torch.cuda.is_available() and properties.get("gpu_id") is not None:
            self.device = torch.device(f"cuda:{properties.get('gpu_id')}")
            logger.info(f"Using GPU: {properties.get('gpu_id')}")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")

        # Load model
        serialized_file = self.manifest["model"]["serializedFile"]
        model_path = os.path.join(model_dir, serialized_file)

        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Load additional config files
        self._load_config(model_dir)

        self.initialized = True
        logger.info("Model initialization complete")

    def _load_config(self, model_dir):
        """Load configuration files"""
        # Class names
        class_file = os.path.join(model_dir, "index_to_name.json")
        if os.path.exists(class_file):
            with open(class_file) as f:
                self.class_names = json.load(f)
            logger.info(f"Loaded class names: {self.class_names}")
        else:
            self.class_names = {"0": "not_churned", "1": "churned"}

        # Feature names
        feature_file = os.path.join(model_dir, "feature_names.json")
        if os.path.exists(feature_file):
            with open(feature_file) as f:
                self.feature_names = json.load(f)
            logger.info(f"Loaded feature names: {self.feature_names}")

    def preprocess(self, data):
        """
        Preprocess input data

        Args:
            data: List of request data

        Returns:
            torch.Tensor: Preprocessed input tensor
        """
        logger.info(f"Preprocessing {len(data)} samples")
        inputs = []

        for row in data:
            # Parse request data
            if isinstance(row, dict):
                features = row.get("data") or row.get("body")
            else:
                features = row.get("body")

            # Handle byte data
            if isinstance(features, (bytes, bytearray)):
                features = json.loads(features.decode("utf-8"))

            # Handle JSON string
            if isinstance(features, str):
                features = json.loads(features)

            # Extract values if dict
            if isinstance(features, dict):
                if "data" in features:
                    features = features["data"]
                else:
                    features = list(features.values())

            # Convert to tensor
            tensor = torch.tensor(features, dtype=torch.float32)
            inputs.append(tensor)

        # Stack into batch
        batch = torch.stack(inputs).to(self.device)
        logger.info(f"Input batch shape: {batch.shape}")

        return batch

    def inference(self, data):
        """
        Run model inference

        Args:
            data: Preprocessed input tensor

        Returns:
            torch.Tensor: Model output
        """
        logger.info("Running inference...")
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model(data)

            # Convert to probabilities (for classification models)
            if outputs.dim() > 1 and outputs.shape[1] > 1:
                probabilities = F.softmax(outputs, dim=1)
            else:
                probabilities = torch.sigmoid(outputs)

        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.4f}s")

        return probabilities

    def postprocess(self, data):
        """
        Postprocess output

        Args:
            data: Model output tensor

        Returns:
            list: JSON-serializable result list
        """
        logger.info("Postprocessing results...")
        results = []

        for prob in data:
            prob_list = prob.cpu().numpy().tolist()

            # Binary classification
            if len(prob_list) == 1:
                prediction = 1 if prob_list[0] > 0.5 else 0
                probabilities = [1 - prob_list[0], prob_list[0]]
            # Multi-class
            else:
                prediction = int(torch.argmax(prob).item())
                probabilities = prob_list

            result = {
                "prediction": prediction,
                "probabilities": probabilities,
                "confidence": max(probabilities)
            }

            # Add class name
            if self.class_names:
                result["class_name"] = self.class_names.get(
                    str(prediction),
                    f"class_{prediction}"
                )

            results.append(result)

        logger.info(f"Processed {len(results)} results")
        return results

    def handle(self, data, context):
        """
        Full request processing (preprocess -> inference -> postprocess)

        Main method called by TorchServe
        """
        if not self.initialized:
            self.initialize(context)

        if data is None:
            return None

        # Preprocess
        model_input = self.preprocess(data)

        # Inference
        model_output = self.inference(model_input)

        # Postprocess
        return self.postprocess(model_output)


# Handler instance (loaded by TorchServe)
_service = ChurnPredictionHandler()


def handle(data, context):
    """TorchServe entry point"""
    return _service.handle(data, context)


# ============================================================
# Local Testing Code
# ============================================================

if __name__ == "__main__":
    import torch.nn as nn

    # Simple test model
    class SimpleModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Create and save model
    print("Creating test model...")
    model = SimpleModel(4, 10, 2)
    model.eval()

    # Save as TorchScript
    scripted = torch.jit.script(model)
    scripted.save("test_model.pt")
    print("Model saved: test_model.pt")

    # Test handler
    print("\nTesting handler...")

    # Mock context
    class MockContext:
        manifest = {"model": {"serializedFile": "test_model.pt"}}
        system_properties = {"model_dir": ".", "gpu_id": None}

    handler = ChurnPredictionHandler()
    handler.initialize(MockContext())

    # Test request
    test_data = [
        {"data": [1.0, 2.0, 3.0, 4.0]},
        {"data": [5.0, 6.0, 7.0, 8.0]}
    ]

    results = handler.handle(test_data, MockContext())

    print("\nResults:")
    for i, result in enumerate(results):
        print(f"  Sample {i+1}:")
        print(f"    Prediction: {result['prediction']}")
        print(f"    Probabilities: {result['probabilities']}")
        print(f"    Confidence: {result['confidence']:.4f}")

    # Cleanup
    import os
    os.remove("test_model.pt")
    print("\nTest complete!")
