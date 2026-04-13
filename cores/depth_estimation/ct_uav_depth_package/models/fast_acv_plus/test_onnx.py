"""
Docstring for ct_uav_depth_package.models.fast_acv_plus.test_onnx
 filepath: /home/thetrung/Documents/CT_UAV/Obstacle_Avoidance/Fast-ACVNet/test_onnx.py
exmaple usage:
python test_onnx.py --model fast_acvnet_plus_generalization_opset16_288x480.onnx \
    --left ./asset/left.png --right ./asset/right.png \
    --output ./out/disparity_onnx.png --maxdisp 192 --use_gpu 
"""


from __future__ import print_function, division
import argparse
import os
import numpy as np
import time
import cv2
from PIL import Image
import onnxruntime as ort

parser = argparse.ArgumentParser(description='Test Fast-ACVNet ONNX model')
parser.add_argument('--model', default='./fast_acv_plus.onnx',
                    help='path to ONNX model')
parser.add_argument('--left', required=True,
                    help='path to left image')
parser.add_argument('--right', required=True,
                    help='path to right image')
parser.add_argument('--output', default='./disparity_onnx.png',
                    help='output disparity image path')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maximum disparity')
parser.add_argument('--use_gpu', action='store_true',
                    help='use GPU for inference (requires CUDA)')

args = parser.parse_args()

def preprocess_image(img_path):
    """
    Load and preprocess image for ONNX model
    Returns: preprocessed tensor and original size
    """
    # Load image
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    
    # Pad to multiple of 32
    wi = (w // 32 + 1) * 32 if w % 32 != 0 else w
    hi = (h // 32 + 1) * 32 if h % 32 != 0 else h
    
    # Crop from bottom-right (same as test_mid.py)
    img = img.crop((w - wi, h - hi, w, h))
    
    # Convert to numpy array
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Normalize with ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_np = (img_np - mean) / std
    
    # Convert to CHW format and add batch dimension
    img_tensor = img_np.transpose(2, 0, 1)
    img_tensor = np.expand_dims(img_tensor, axis=0).astype(np.float32)
    
    return img_tensor, (w, h), (wi, hi)

def postprocess_disparity(disp, orig_size, padded_size):
    """
    Postprocess disparity map
    """
    w, h = orig_size
    wi, hi = padded_size
    
    # Crop to original size
    disp = disp[hi - h:, wi - w:]
    
    return disp

def visualize_disparity(disp, output_path, maxdisp=192):
    """
    Visualize and save disparity map
    """
    # Normalize to 0-255
    disp_vis = (disp / maxdisp * 255.0).astype(np.uint8)
    
    # Apply colormap
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    
    # Save
    cv2.imwrite(output_path, disp_color)
    
    # Also save raw disparity
    raw_output = output_path.replace('.png', '_raw.png')
    disp_16bit = (disp * 256).astype(np.uint16)
    cv2.imwrite(raw_output, disp_16bit, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    
    return disp_color

def test_onnx():
    print("=" * 50)
    print("Testing Fast-ACVNet ONNX model")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Check if images exist
    if not os.path.exists(args.left):
        print(f"Error: Left image not found: {args.left}")
        return
    if not os.path.exists(args.right):
        print(f"Error: Right image not found: {args.right}")
        return
    
    # Load ONNX model
    print(f"Loading ONNX model: {args.model}")
    
    providers = ['CPUExecutionProvider']
    if args.use_gpu:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    session = ort.InferenceSession(args.model, providers=providers)
    
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"Using providers: {session.get_providers()}")
    
    # Print model info
    print("\nModel Inputs:")
    for input_tensor in session.get_inputs():
        print(f"  {input_tensor.name}: {input_tensor.shape} ({input_tensor.type})")
    
    print("\nModel Outputs:")
    for output_tensor in session.get_outputs():
        print(f"  {output_tensor.name}: {output_tensor.shape} ({output_tensor.type})")
    
    # Preprocess images
    print(f"\nLoading and preprocessing images...")
    print(f"  Left image: {args.left}")
    print(f"  Right image: {args.right}")
    
    left_tensor, orig_size, padded_size = preprocess_image(args.left)
    right_tensor, _, _ = preprocess_image(args.right)
    
    print(f"  Original size: {orig_size}")
    print(f"  Padded size: {padded_size}")
    print(f"  Input tensor shape: {left_tensor.shape}")
    
    # Run inference multiple times for timing
    print("\nRunning inference (10 runs for timing)...")

    input_name_left = session.get_inputs()[0].name
    input_name_right = session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name

    times = []
    outputs = None
    for i in range(10):
        start_time = time.time()
        outputs = session.run(
            [output_name],
            {input_name_left: left_tensor, input_name_right: right_tensor}
        )
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"  Run {i + 1:02d}: {elapsed * 1000:.2f} ms")

    times = np.array(times, dtype=np.float32)
    print("\nInference time statistics over 10 runs:")
    print(f"  Avg: {times.mean() * 1000:.2f} ms")
    print(f"  Min: {times.min() * 1000:.2f} ms")
    print(f"  Max: {times.max() * 1000:.2f} ms")
    print(f"  Output shape (last run): {outputs[0].shape}")
    
    # Postprocess disparity
    disp = outputs[0].squeeze()
    disp = postprocess_disparity(disp, orig_size, padded_size)
    
    print(f"\nDisparity statistics:")
    print(f"  Min: {disp.min():.2f}")
    print(f"  Max: {disp.max():.2f}")
    print(f"  Mean: {disp.mean():.2f}")
    print(f"  Std: {disp.std():.2f}")
    
    # Visualize and save
    print(f"\nSaving disparity to: {args.output}")
    disp_color = visualize_disparity(disp, args.output, args.maxdisp)
    
    raw_output = args.output.replace('.png', '_raw.png')
    print(f"Saving raw disparity to: {raw_output}")
    
    print("=" * 50)
    print("✓ Test completed successfully!")
    print("=" * 50)

if __name__ == '__main__':
    test_onnx()
