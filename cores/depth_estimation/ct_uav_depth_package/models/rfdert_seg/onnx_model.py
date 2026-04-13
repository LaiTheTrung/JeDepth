import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort
import supervision as sv


class ONNXSegmentationModel:
    """ONNX-based segmentation model wrapper"""
    
    def __init__(self, onnx_path, use_gpu=True, input_size=576):
        """
        Initialize ONNX model
        
        Args:
            onnx_path: Path to ONNX model file
            use_gpu: Use CUDA if available
            input_size: Model input size (default: 576)
        """
        self.input_size = input_size
        
        # Mean and std for normalization (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Setup ONNX Runtime
        providers = []
        if use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
    
    def preprocess(self, image):
        """
        Preprocess image for inference
        
        Args:
            image: numpy array (H, W, 3) in BGR format
            
        Returns:
            tuple: (preprocessed_array, original_size)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original size
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # Resize to model input size
        image_resized = cv2.resize(
            image_rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize
        img_array = image_resized.astype(np.float32) / 255.0
        img_array = (img_array - self.mean.reshape(1, 1, 3)) / self.std.reshape(1, 1, 3)
        
        # Transpose to (C, H, W) and add batch dimension
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        return img_array, original_size
    
    def postprocess(self, outputs, original_size, threshold):
        """
        Postprocess model outputs to detections
        
        Args:
            outputs: Model output tensors
            original_size: (width, height) of original image
            threshold: Confidence threshold
            
        Returns:
            sv.Detections: Detections with masks
        """
        # Extract outputs: [dets, labels, masks]
        dets = outputs[0][0]  # (N, 4) bounding boxes
        labels = outputs[1][0]  # (N, num_classes) class probabilities
        masks = outputs[2][0]  # (N, H, W) segmentation masks
        
        # Get scores and class IDs
        scores = labels.max(axis=1)
        class_ids = labels.argmax(axis=1)
        
        # Filter by threshold
        valid_indices = scores > threshold
        dets = dets[valid_indices]
        scores = scores[valid_indices]
        class_ids = class_ids[valid_indices]
        masks = masks[valid_indices]
        
        # Return empty if no detections
        if len(dets) == 0:
            return sv.Detections.empty()
        
        # Scale bounding boxes to original size
        scale_x = original_size[0] / self.input_size
        scale_y = original_size[1] / self.input_size
        
        dets = dets.astype(np.float32)
        dets[:, [0, 2]] *= scale_x
        dets[:, [1, 3]] *= scale_y
        
        # Resize masks to original size
        resized_masks = []
        for mask in masks:
            mask_resized = cv2.resize(
                mask.astype(np.float32),
                original_size,
                interpolation=cv2.INTER_LINEAR
            )
            mask_binary = (mask_resized > 0.5).astype(bool)
            resized_masks.append(mask_binary)
        
        # Create supervision Detections object
        detections = sv.Detections(
            xyxy=dets,
            confidence=scores,
            class_id=class_ids.astype(int),
            mask=np.array(resized_masks) if resized_masks else None
        )
        
        return detections
    
    def predict(self, image, threshold=0.5):
        """
        Run inference on image
        
        Args:
            image: numpy array (H, W, 3) in BGR format
            threshold: Confidence threshold
            
        Returns:
            sv.Detections: Detections with masks
        """
        # Preprocess
        input_data, original_size = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_data}
        )
        
        # Postprocess
        detections = self.postprocess(outputs, original_size, threshold)
        
        return detections
