#!/usr/bin/env python3
"""
ONNX Runtime-based Edge AI Inference
Image classification and object detection examples

Reference: content/ko/IoT_Embedded/09_Edge_AI_ONNX.md
"""

import numpy as np
import time
from typing import Optional, Tuple, List, Dict
import os

# Check ONNX Runtime installation
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: onnxruntime is not installed.")
    print("Install: pip install onnxruntime")

# Check OpenCV installation
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: opencv-python is not installed.")
    print("Install: pip install opencv-python")

# Check PIL installation
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: Pillow is not installed.")
    print("Install: pip install Pillow")


# === ONNX Model Wrapper ===

class ONNXModel:
    """ONNX Model Base Wrapper"""

    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        if not HAS_ONNX:
            raise ImportError("onnxruntime is required")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Why: ONNX Runtime selects the execution provider (CPU, CUDA, TensorRT)
        # at session creation time. Auto-detecting available providers lets the
        # same code run on a GPU server or a CPU-only Raspberry Pi without changes.
        if providers is None:
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

        # Why: ORT_ENABLE_ALL applies constant folding, operator fusion, and
        # memory planning at load time. This one-time cost yields ~15-30% speedup
        # across all subsequent inferences — critical for edge latency budgets.
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        # Create session
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        # Input/output information
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_type = self.session.get_inputs()[0].type
        self.output_name = self.session.get_outputs()[0].name

        print(f"Model loaded: {model_path}")
        print(f"  Provider: {self.session.get_providers()}")
        print(f"  Input: {self.input_name} {self.input_shape}")
        print(f"  Output: {self.output_name}")

    def get_input_shape(self) -> list:
        """Return input shape"""
        return self.input_shape

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Perform inference"""
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        return outputs[0]

    def benchmark(self, num_iterations: int = 100) -> Dict[str, float]:
        """Performance benchmark"""
        # Create dummy input
        dummy_shape = [1 if x == 'batch' or x == 'N' or x is None else x
                      for x in self.input_shape]
        dummy_input = np.random.randn(*dummy_shape).astype(np.float32)

        # Warmup
        for _ in range(10):
            self.predict(dummy_input)

        # Measurement
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.predict(dummy_input)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        times = np.array(times)

        results = {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "fps": 1000.0 / np.mean(times)
        }

        return results


# === Image Classification Model ===

class ImageClassifier(ONNXModel):
    """ONNX Image Classification Model"""

    # ImageNet classes (top 10 only as example)
    IMAGENET_CLASSES = [
        'tench', 'goldfish', 'great_white_shark', 'tiger_shark',
        'hammerhead', 'electric_ray', 'stingray', 'cock', 'hen', 'ostrich'
        # ... actually 1000 classes
    ]

    def __init__(self, model_path: str, labels_path: Optional[str] = None):
        super().__init__(model_path)

        # Load labels (if available)
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f]
        else:
            self.labels = self.IMAGENET_CLASSES

        # Extract input size
        self.input_height = self.input_shape[2] if len(self.input_shape) > 2 else 224
        self.input_width = self.input_shape[3] if len(self.input_shape) > 3 else 224

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Image preprocessing (using PIL)"""
        if not HAS_PIL:
            raise ImportError("Pillow is required")

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Resize
        image = image.resize((self.input_width, self.input_height))

        # Convert to NumPy array
        img_array = np.array(image).astype(np.float32)

        # Why: ImageNet-trained models (ResNet, EfficientNet, etc.) were trained
        # with these exact mean/std values. Skipping normalization or using
        # different values will produce meaningless softmax outputs.
        mean = np.array([0.485, 0.456, 0.406]) * 255
        std = np.array([0.229, 0.224, 0.225]) * 255
        img_array = (img_array - mean) / std

        # Why: PyTorch-exported ONNX models expect CHW layout (Channels, Height,
        # Width), but PIL loads images as HWC. This transpose is a silent requirement.
        img_array = img_array.transpose(2, 0, 1)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def classify(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Image classification"""
        # Preprocessing
        input_data = self.preprocess_image(image_path)

        # Inference
        start = time.perf_counter()
        output = self.predict(input_data)
        inference_time = (time.perf_counter() - start) * 1000

        # Softmax
        probs = self._softmax(output[0])

        # Top-K results
        top_indices = np.argsort(probs)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
            results.append((label, float(probs[idx])))

        print(f"Inference time: {inference_time:.2f}ms")

        return results

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


# === Object Detection Model (YOLO) ===

class YOLODetector:
    """YOLO ONNX Object Detector"""

    # COCO dataset 80 classes
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(self, model_path: str, conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        if not HAS_ONNX:
            raise ImportError("onnxruntime is required")

        if not os.path.exists(model_path):
            # If model not found, run in simulation mode
            print(f"Warning: Model file not found: {model_path}")
            print("Running in simulation mode.")
            self.simulation_mode = True
            self.input_height = 640
            self.input_width = 640
            return

        self.simulation_mode = False

        # Create ONNX session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Input information
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_shape = input_info.shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        print(f"YOLO model loaded")
        print(f"  Input size: {self.input_width}x{self.input_height}")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Image preprocessing"""
        orig_height, orig_width = image.shape[:2]

        # Resize
        resized = cv2.resize(image, (self.input_width, self.input_height))

        # BGR to RGB, HWC to CHW
        input_data = resized[:, :, ::-1].transpose(2, 0, 1)

        # Normalize (0-1)
        input_data = input_data.astype(np.float32) / 255.0

        # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)

        # Save scale ratio
        scale = (orig_width / self.input_width, orig_height / self.input_height)

        return input_data, scale

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Object detection"""
        if self.simulation_mode:
            # Simulation: return random detection results
            print("Simulation mode: generating random detection results.")
            return self._simulate_detection(image)

        if not HAS_CV2:
            raise ImportError("opencv-python is required")

        # Preprocessing
        input_data, scale = self.preprocess(image)

        # Inference
        start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: input_data})
        inference_time = (time.perf_counter() - start) * 1000

        # Post-processing
        detections = self.postprocess(outputs[0], scale)

        print(f"Inference time: {inference_time:.2f}ms")
        print(f"Detected objects: {len(detections)}")

        return detections

    def postprocess(self, output: np.ndarray, scale: Tuple[float, float]) -> List[Dict]:
        """Output post-processing"""
        if not HAS_CV2:
            return []

        predictions = output[0]

        boxes = []
        scores = []
        class_ids = []

        for pred in predictions:
            confidence = pred[4]

            if confidence > self.conf_threshold:
                class_probs = pred[5:]
                class_id = np.argmax(class_probs)
                class_score = class_probs[class_id]

                if class_score > self.conf_threshold:
                    # Box coordinates (center_x, center_y, width, height)
                    cx, cy, w, h = pred[:4]

                    # Convert to original scale
                    x1 = int((cx - w / 2) * scale[0])
                    y1 = int((cy - h / 2) * scale[1])
                    x2 = int((cx + w / 2) * scale[0])
                    y2 = int((cy + h / 2) * scale[1])

                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(confidence * class_score))
                    class_ids.append(int(class_id))

        # Why: Without NMS, overlapping detections of the same object produce
        # duplicate bounding boxes. NMS keeps only the highest-confidence box
        # among those with IoU above the threshold.
        if boxes:
            indices = cv2.dnn.NMSBoxes(
                boxes, scores, self.conf_threshold, self.iou_threshold
            )

            results = []
            for i in indices:
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                results.append({
                    'box': boxes[idx],
                    'score': scores[idx],
                    'class_id': class_ids[idx],
                    'class_name': self.COCO_CLASSES[class_ids[idx]]
                })

            return results

        return []

    def _simulate_detection(self, image: np.ndarray) -> List[Dict]:
        """Simulation: random detection results"""
        height, width = image.shape[:2]

        num_detections = np.random.randint(1, 5)
        detections = []

        for _ in range(num_detections):
            x1 = np.random.randint(0, width // 2)
            y1 = np.random.randint(0, height // 2)
            x2 = np.random.randint(x1 + 50, width)
            y2 = np.random.randint(y1 + 50, height)

            class_id = np.random.randint(0, len(self.COCO_CLASSES))

            detections.append({
                'box': [x1, y1, x2, y2],
                'score': np.random.uniform(0.5, 0.95),
                'class_id': class_id,
                'class_name': self.COCO_CLASSES[class_id]
            })

        return detections

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Visualize detection results"""
        if not HAS_CV2:
            print("Warning: opencv-python is not available, skipping visualization.")
            return image

        result = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = f"{det['class_name']}: {det['score']:.2f}"

            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)

            # Label text
            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return result


# === Usage Examples ===

def example_basic_inference():
    """Basic ONNX inference example"""
    print("\n=== Basic ONNX Inference Example ===")

    if not HAS_ONNX:
        print("Cannot run example: onnxruntime is not installed.")
        return

    # Simulation: create dummy model
    print("Simulation mode: testing with dummy input.")

    # Dummy data
    batch_size = 1
    channels = 3
    height = 224
    width = 224

    dummy_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Input data range: [{dummy_input.min():.2f}, {dummy_input.max():.2f}]")


def example_image_classification():
    """Image classification example"""
    print("\n=== Image Classification Example ===")

    if not HAS_ONNX:
        print("Cannot run example: onnxruntime is not installed.")
        return

    # Model path (example)
    model_path = "resnet18.onnx"

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Conversion example from PyTorch:")
        print("  import torch")
        print("  model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)")
        print("  dummy_input = torch.randn(1, 3, 224, 224)")
        print("  torch.onnx.export(model, dummy_input, 'resnet18.onnx')")
        return

    # Create classifier
    classifier = ImageClassifier(model_path)

    # Benchmark
    print("\nPerformance benchmark:")
    results = classifier.benchmark(num_iterations=50)
    print(f"  Average: {results['mean_ms']:.2f}ms")
    print(f"  FPS: {results['fps']:.1f}")


def example_object_detection():
    """Object detection example"""
    print("\n=== Object Detection Example (Simulation) ===")

    # Run in simulation mode
    detector = YOLODetector("yolov5s.onnx")  # Works even without the file

    # Create dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Detection
    detections = detector.detect(dummy_image)

    # Print results
    print("\nDetection results:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class_name']}: {det['score']:.2f}")
        print(f"     Box: {det['box']}")

    # Visualization (if OpenCV is available)
    if HAS_CV2:
        result_image = detector.draw_detections(dummy_image, detections)
        print("\nResult image generated")


def example_performance_comparison():
    """Performance comparison example"""
    print("\n=== Performance Comparison Example ===")

    if not HAS_ONNX:
        print("Cannot run example: onnxruntime is not installed.")
        return

    print("Performance comparison by batch size (simulation)")

    input_shape = (1, 3, 224, 224)

    for batch_size in [1, 4, 8, 16]:
        data = np.random.randn(batch_size, *input_shape[1:]).astype(np.float32)

        start = time.perf_counter()
        # Simulation: simple computation
        _ = np.mean(data, axis=(2, 3))
        elapsed = time.perf_counter() - start

        throughput = batch_size / elapsed
        print(f"Batch size {batch_size:2d}: {throughput:.1f} samples/sec")


# === Main Execution ===

if __name__ == "__main__":
    print("=" * 60)
    print("ONNX Runtime Edge AI Inference Example")
    print("=" * 60)

    # Check ONNX Runtime installation
    if HAS_ONNX:
        print(f"\nONNX Runtime version: {ort.__version__}")
        print(f"Available providers: {ort.get_available_providers()}")
    else:
        print("\nWarning: ONNX Runtime is not installed.")
        print("Install: pip install onnxruntime")

    # Run examples
    example_basic_inference()
    example_image_classification()
    example_object_detection()
    example_performance_comparison()

    print("\n" + "=" * 60)
    print("All examples completed")
    print("=" * 60)
