"""
Fast-ACVNet TensorRT inference module
Based on ONNX inference logic from test_onnx.py
"""

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import List, Tuple
import os
from PIL import Image

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
print(f"TensorRT Version: {trt.__version__}")


def _get_tensor_index(engine: trt.ICudaEngine, name: str) -> int:
    """Get tensor index by name for TensorRT 10.x tensor-based API."""
    num_tensors = engine.num_io_tensors
    for i in range(num_tensors):
        t_name = engine.get_tensor_name(i)
        if t_name == name:
            return i
    raise ValueError(f"Tensor '{name}' not found in engine")


class FastACVExecutor:
    """Single executor for Fast-ACVNet inference on one stream"""
    
    def __init__(self, engine, stream, input_left_name: str = "left_image", 
                 input_right_name: str = "right_image",
                 output_name: str = "output"):
        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = stream
        self.input_left_name = input_left_name
        self.input_right_name = input_right_name
        self.output_name = output_name

        # Get tensor indices (TensorRT 10.x tensor-based API)
        self.input_left_idx = _get_tensor_index(engine, input_left_name)
        self.input_right_idx = _get_tensor_index(engine, input_right_name)
        self.output_idx = _get_tensor_index(engine, output_name)

        # Get input/output shapes
        self.input_left_shape = engine.get_tensor_shape(self.input_left_name)
        self.input_right_shape = engine.get_tensor_shape(self.input_right_name)
        self.output_shape = engine.get_tensor_shape(self.output_name)
        
        # Calculate sizes
        self.input_left_size = trt.volume(self.input_left_shape) * np.dtype(np.float32).itemsize
        self.input_right_size = trt.volume(self.input_right_shape) * np.dtype(np.float32).itemsize
        self.output_size = trt.volume(self.output_shape) * np.dtype(np.float32).itemsize
        
        # Allocate device memory
        self.d_input_left = cuda.mem_alloc(self.input_left_size)
        self.d_input_right = cuda.mem_alloc(self.input_right_size)
        self.d_output = cuda.mem_alloc(self.output_size)
        
        # Allocate host memory
        self.h_input_left = cuda.pagelocked_empty(trt.volume(self.input_left_shape), dtype=np.float32)
        self.h_input_right = cuda.pagelocked_empty(trt.volume(self.input_right_shape), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)
        
        # Bindings array for execute_async_v3
        num_tensors = engine.num_io_tensors
        self.bindings = [0] * num_tensors
        self.bindings[self.input_left_idx] = int(self.d_input_left)
        self.bindings[self.input_right_idx] = int(self.d_input_right)
        self.bindings[self.output_idx] = int(self.d_output)
        
        print(f"FastACVExecutor initialized")
        print(f"  Input Left: {self.input_left_shape}")
        print(f"  Input Right: {self.input_right_shape}")
        print(f"  Output: {self.output_shape}")
    
    def set_input(self, left_tensor: np.ndarray, right_tensor: np.ndarray) -> int:
        """Set input left and right images (already preprocessed)"""
        try:
            np.copyto(self.h_input_left, left_tensor.ravel())
            np.copyto(self.h_input_right, right_tensor.ravel())
            return 0
        except Exception as e:
            print(f"Error setting input: {e}")
            return -1
    
    def infer(self) -> int:
        """Run inference"""
        try:
            # Copy inputs to device
            cuda.memcpy_htod_async(self.d_input_left, self.h_input_left, self.stream)
            cuda.memcpy_htod_async(self.d_input_right, self.h_input_right, self.stream)

            # Set shapes if dynamic (TensorRT 10.x tensor API)
            try:
                self.context.set_input_shape(self.input_left_name, self.input_left_shape)
                self.context.set_input_shape(self.input_right_name, self.input_right_shape)
            except Exception:
                # If engine is static shape, this may fail; ignore
                pass

            # Execute using TensorRT 10.x API
            try:
                # Register tensor addresses
                if hasattr(self.context, "set_tensor_address"):
                    self.context.set_tensor_address(self.input_left_name, int(self.d_input_left))
                    self.context.set_tensor_address(self.input_right_name, int(self.d_input_right))
                    self.context.set_tensor_address(self.output_name, int(self.d_output))
                self.context.execute_async_v3(self.stream.handle)
            except TypeError:
                # Fallback: older signature may still accept bindings list
                self.context.execute_async_v3(self.stream.handle)

            # Copy output to host
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            
            return 0
        except Exception as e:
            print(f"Error during inference: {e}")
            return -1
    
    def synchronize(self) -> int:
        """Synchronize stream"""
        try:
            self.stream.synchronize()
            return 0
        except Exception as e:
            print(f"Error synchronizing: {e}")
            return -1
    
    def get_output(self) -> np.ndarray:
        """Get output disparity map"""
        return self.h_output.reshape(self.output_shape)


class FastACVTRT:
    """Fast-ACVNet TensorRT inference for multiple stereo pairs"""
    
    def __init__(self, show_info: bool = True):
        self.show_info = show_info
        self.engine = None
        self.executors: List[FastACVExecutor] = []
        self.streams: List[cuda.Stream] = []
        self.stream_number = 0
        
    def init(self, onnx_path: str, engine_path: str, stream_number: int = 4) -> int:
        """Initialize TensorRT engine and executors"""
        if stream_number <= 0:
            stream_number = 1
        self.stream_number = stream_number
        
        # Load or build engine
        if not os.path.exists(engine_path):
            print(f"Engine file not found at {engine_path}, building from ONNX...")
            if not os.path.exists(onnx_path):
                print(f"ONNX file not found at {onnx_path}")
                return -1
            if self._build_engine(onnx_path, engine_path) != 0:
                print("Failed to build engine")
                return -2
        else:
            print(f"Loading engine from {engine_path}")
            if self._load_engine(engine_path) != 0:
                print("Failed to load engine")
                return -3
        
        # Create streams and executors
        for i in range(self.stream_number):
            stream = cuda.Stream()
            self.streams.append(stream)
            
            executor = FastACVExecutor(self.engine, stream)
            self.executors.append(executor)
        
        print(f"FastACVTRT initialized with {self.stream_number} streams")
        return 0
    
    def _load_engine(self, engine_path: str) -> int:
        """Load TensorRT engine from file"""
        try:
            with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            if self.engine is None:
                print("Failed to deserialize engine")
                return -1
            
            print("Engine loaded successfully")
            return 0
        except Exception as e:
            print(f"Error loading engine: {e}")
            return -1
    
    def _build_engine(self, onnx_path: str, engine_path: str) -> int:
        """Build TensorRT engine from ONNX model"""
        try:
            '''Build by command line'''
            import subprocess
            model_parent = os.path.dirname(onnx_path)
            onnx_model_name = os.path.basename(onnx_path).split(".")[0]
            new_engine_path = os.path.basename(engine_path)
            print(f"Building engine for model: {onnx_model_name} in {model_parent}")
            cmd = f"cd {model_parent} && trtexec --onnx={onnx_model_name}.onnx --saveEngine={new_engine_path} --fp16 --memPoolSize=workspace:4096 --verbose"
            print(f"Building engine with command: {cmd}")
            print("This may take several minutes...")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error building engine: {result.stderr}")
                return -1
            print("Engine built and saved successfully")
            print("Engine path:", os.path.join(model_parent, new_engine_path))
            return 0
        except Exception as e:
            print(f"Error building engine: {e}")
            return -1
    
    def preprocess(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple, Tuple]:
        """
        Preprocess stereo images into input format
        Based on test_onnx.py preprocessing logic
        Returns: (left_tensor, right_tensor, orig_size, padded_size)
        """
        if left_img is None or right_img is None:
            return None, None, None, None
        
        if left_img.shape != right_img.shape:
            print("Left and right images must have the same shape")
            return None, None, None, None
        
        # Convert BGR to RGB if needed
        if len(left_img.shape) == 3 and left_img.shape[2] == 3:
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        
        h, w = left_img.shape[:2]
        orig_size = (w, h)
        
        # Pad to multiple of 32
        wi = (w // 32 + 1) * 32 if w % 32 != 0 else w
        hi = (h // 32 + 1) * 32 if h % 32 != 0 else h
        padded_size = (wi, hi)
        
        # Crop from bottom-right (same as test_onnx.py)
        left_cropped = left_img[h - hi:, w - wi:]
        right_cropped = right_img[h - hi:, w - wi:]
        
        # Convert to float32 and normalize to [0, 1]
        left_np = left_cropped.astype(np.float32) / 255.0
        right_np = right_cropped.astype(np.float32) / 255.0
        
        # Normalize with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        left_np = (left_np - mean) / std
        right_np = (right_np - mean) / std
        
        # Convert to CHW format and add batch dimension
        left_tensor = left_np.transpose(2, 0, 1)
        right_tensor = right_np.transpose(2, 0, 1)
        left_tensor = np.expand_dims(left_tensor, axis=0).astype(np.float32)
        right_tensor = np.expand_dims(right_tensor, axis=0).astype(np.float32)
        
        return left_tensor, right_tensor, orig_size, padded_size
    
    def postprocess_disparity(self, disp: np.ndarray, orig_size: Tuple, padded_size: Tuple) -> np.ndarray:
        """
        Postprocess disparity map
        Based on test_onnx.py postprocessing logic
        """
        w, h = orig_size
        wi, hi = padded_size
        
        # Remove batch dimension if present
        if len(disp.shape) == 3 and disp.shape[0] == 1:
            disp = disp.squeeze(0)
        if len(disp.shape) == 3 and disp.shape[0] == 1:
            disp = disp.squeeze(0)
        
        # Crop to original size
        disp = disp[hi - h:, wi - w:]
        
        return disp
    
    def do_inference(self, left_images: List[np.ndarray], right_images: List[np.ndarray]) -> int:
        """
        Run inference on multiple stereo pairs
        left_images, right_images: Lists of preprocessed tensors
        """
        if len(left_images) != self.stream_number or len(right_images) != self.stream_number:
            print(f"Expected {self.stream_number} image pairs, got {len(left_images)} left and {len(right_images)} right")
            return -1
        
        # Set inputs for all executors
        for i, (left, right) in enumerate(zip(left_images, right_images)):
            if left is None or right is None:
                continue
            
            if self.executors[i].set_input(left, right) != 0:
                return -2
        
        # Execute inference asynchronously
        for executor in self.executors:
            if executor.infer() != 0:
                return -3
        
        # Synchronize all streams
        for executor in self.executors:
            if executor.synchronize() != 0:
                return -4
        
        return 0
    
    def get_output(self) -> List[np.ndarray]:
        """Get output disparity maps from all executors"""
        outputs = []
        for executor in self.executors:
            output = executor.get_output()
            outputs.append(output)
        return outputs
    
    def visualize_disparity(self, disp: np.ndarray, maxdisp: int = 192) -> np.ndarray:
        """
        Visualize disparity map with colormap
        Based on test_onnx.py visualization logic
        """
        # Normalize to 0-255
        if len(disp.shape) >= 2:
            if len(disp.shape) == 3 and disp.shape[0] == 1:
                disp = disp[0]
            elif len(disp.shape) == 4:
                disp = disp[0,0]
        print(f"Visualizing disparity with shape: {disp.shape}")
        disp_vis = (disp / maxdisp * 255.0).astype(np.uint8)
        
        # Apply colormap
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        
        return disp_color
    
    def __del__(self):
        """Cleanup"""
        if self.engine:
            del self.engine
