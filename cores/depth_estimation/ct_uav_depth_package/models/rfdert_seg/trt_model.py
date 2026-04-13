import torch
import tensorrt as trt
import numpy as np
import cv2
import supervision as sv

class TRTSegmentationModel:
    """TensorRT-based segmentation model wrapper using PyTorch backend"""
    
    def __init__(self, engine_path, input_size=576):
        """
        Initialize TensorRT model
        
        Args:
            engine_path: Path to TensorRT engine file
            input_size: Model input size (default: 576)
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.input_size = input_size
        
        # Mean and std for normalization (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Check CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! TensorRT requires CUDA.")
            
        self.device = torch.device("cuda:0")
        
        # Load Engine
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        if not self.engine:
            raise RuntimeError("Failed to load TensorRT engine")

        self.context = self.engine.create_execution_context()
        
        # Allocate Buffers using PyTorch
        self.bindings = {}      # Dict to hold PyTorch tensors
        self.inputs = []        # List of input names
        self.outputs = []       # List of output names
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            
            # Handle Dynamic Shape (-1)
            if -1 in shape:
                lst_shape = list(shape)
                for idx, dim in enumerate(lst_shape):
                    if dim == -1: lst_shape[idx] = 1
                shape = tuple(lst_shape)
                
                if mode == trt.TensorIOMode.INPUT:
                    self.context.set_input_shape(name, shape)

            # Create Tensor on GPU
            tensor = torch.zeros(tuple(shape), dtype=torch.float32, device=self.device)
            
            self.bindings[name] = tensor
            
            # Map memory address for TensorRT Context
            self.context.set_tensor_address(name, tensor.data_ptr())

            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)

        # PyTorch stream for synchronization
        self.stream = torch.cuda.current_stream().cuda_stream

    def preprocess(self, image):
        """
        Preprocess image for inference
        
        Args:
            image: numpy array (H, W, 3) in BGR format
            
        Returns:
            tuple: (original_size)
        """
        # Store original size
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image_resized = cv2.resize(
            image_rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize
        img_array = image_resized.astype(np.float32) / 255.0
        img_array = (img_array - self.mean) / self.std
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Copy numpy -> GPU Tensor
        input_name = self.inputs[0] # Assume single input
        self.bindings[input_name].copy_(torch.from_numpy(img_array))
        
        return original_size

    def infer(self):
        """Execute inference"""
        # Execute (Async V3) using PyTorch stream
        self.context.execute_async_v3(stream_handle=self.stream)
        
        # Synchronize
        torch.cuda.synchronize()
        
        # Return results as Numpy (Copy GPU -> CPU)
        return {name: self.bindings[name].cpu().numpy() for name in self.outputs}

    def postprocess(self, outputs_dict, original_size, threshold):
        """
        Postprocess model outputs
        
        Args:
            outputs_dict: Dictionary of output arrays
            original_size: (width, height)
            threshold: Confidence threshold
            
        Returns:
            sv.Detections: Detections object
        """
        dets, labels, masks = None, None, None
        
        # Auto-detect based on shape
        for name, arr in outputs_dict.items():
            shape = arr.shape
            if len(shape) == 3 and shape[-1] == 4:
                dets = arr[0]
            elif len(shape) == 3 and shape[-1] > 4:
                labels = arr[0]
            elif len(shape) == 4:
                masks = arr[0]
                
        # Fallback if auto-detect fails
        if dets is None:
            vals = list(outputs_dict.values())
            dets, labels, masks = vals[0][0], vals[1][0], vals[2][0]

        # Process scores and classes
        scores = labels.max(axis=1)
        class_ids = labels.argmax(axis=1)
        
        # Filter by threshold
        valid_indices = scores > threshold
        
        dets = dets[valid_indices]
        scores = scores[valid_indices]
        class_ids = class_ids[valid_indices]
        masks = masks[valid_indices]
        
        if len(dets) == 0:
            return sv.Detections.empty()
        
        # Scale bounding boxes
        scale_x = original_size[0] / self.input_size
        scale_y = original_size[1] / self.input_size
        
        dets = dets.copy()
        dets[:, [0, 2]] *= scale_x
        dets[:, [1, 3]] *= scale_y
        
        # Resize masks
        resized_masks = []
        for mask in masks:
            mask_resized = cv2.resize(
                mask, 
                original_size, 
                interpolation=cv2.INTER_LINEAR
            )
            resized_masks.append(mask_resized > 0.5)
            
        return sv.Detections(
            xyxy=dets,
            confidence=scores,
            class_id=class_ids.astype(int),
            mask=np.array(resized_masks) if resized_masks else None
        )

    def predict(self, image, threshold=0.5):
        """
        Run full prediction pipeline
        
        Args:
            image: numpy array (H, W, 3) in BGR format
            threshold: Confidence threshold
            
        Returns:
            sv.Detections: Detections object
        """
        original_size = self.preprocess(image)
        outputs = self.infer()
        return self.postprocess(outputs, original_size, threshold)
