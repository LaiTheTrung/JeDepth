"""
Test Fast-ACVNet TensorRT inference
Based on test_onnx.py logic with TensorRT backend
"""

import argparse
import numpy as np
import cv2
import time
from fastACV import FastACVTRT


def test_trt():
    parser = argparse.ArgumentParser(description='Test Fast-ACVNet TensorRT inference')
    parser.add_argument('--onnx', default='./fast_acvnet_plus_generalization_opset16_288x480.onnx',
                        help='path to ONNX model')
    parser.add_argument('--engine', default='./fast_acv_plus.engine',
                        help='path to TensorRT engine')
    parser.add_argument('--left', required=True,
                        help='path to left image')
    parser.add_argument('--right', required=True,
                        help='path to right image')
    parser.add_argument('--output', default='./disparity_trt.png',
                        help='output disparity image path')
    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maximum disparity')
    parser.add_argument('--streams', type=int, default=1,
                        help='number of streams')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Testing Fast-ACVNet TensorRT model")
    print("=" * 50)
    
    # Initialize TensorRT model
    print(f"\nInitializing TensorRT model...")
    print(f"  ONNX: {args.onnx}")
    print(f"  Engine: {args.engine}")
    print(f"  Streams: {args.streams}")
    
    model = FastACVTRT(show_info=True)
    ret = model.init(args.onnx, args.engine, stream_number=args.streams)
    if ret != 0:
        print(f"Failed to initialize model, error code: {ret}")
        exit(1)
    
    # Load images
    print(f"\nLoading images...")
    print(f"  Left: {args.left}")
    print(f"  Right: {args.right}")
    
    left_img = cv2.imread(args.left)
    right_img = cv2.imread(args.right)
    
    if left_img is None or right_img is None:
        print("Failed to load images")
        exit(1)
    
    print(f"  Image shape: {left_img.shape}")
    
    # Preprocess
    print(f"\nPreprocessing images...")
    left_tensor, right_tensor, orig_size, padded_size = model.preprocess(left_img, right_img)
    
    if left_tensor is None or right_tensor is None:
        print("Failed to preprocess images")
        exit(1)
    
    print(f"  Original size: {orig_size}")
    print(f"  Padded size: {padded_size}")
    print(f"  Tensor shapes: Left={left_tensor.shape}, Right={right_tensor.shape}")
    
    # Prepare inputs for all streams (duplicate for testing)
    left_inputs = [left_tensor] * args.streams
    right_inputs = [right_tensor] * args.streams
    
    # Run inference multiple times for timing
    print("\nRunning inference (10 runs for timing)...")
    times = []
    for i in range(10):
        start_time = time.time()
        ret = model.do_inference(left_inputs, right_inputs)
        if ret != 0:
            print(f"Inference failed with error code: {ret}")
            exit(1)
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"  Run {i + 1:02d}: {elapsed * 1000:.2f} ms")
    
    times = np.array(times)
    print(f"\nInference time statistics over 10 runs:")
    print(f"  Avg: {times.mean() * 1000:.2f} ms")
    print(f"  Min: {times.min() * 1000:.2f} ms")
    print(f"  Max: {times.max() * 1000:.2f} ms")
    
    # Get output
    outputs = model.get_output()
    disp = outputs[0]  # Get first output
    
    # Postprocess
    disp = model.postprocess_disparity(disp, orig_size, padded_size)
    
    print(f"\nDisparity statistics:")
    print(f"  Shape: {disp.shape}")
    print(f"  Min: {disp.min():.2f}")
    print(f"  Max: {disp.max():.2f}")
    print(f"  Mean: {disp.mean():.2f}")
    print(f"  Std: {disp.std():.2f}")
    
    # Visualize and save
    print(f"\nSaving disparity to: {args.output}")
    disp_color = model.visualize_disparity(disp, args.maxdisp)
    cv2.imwrite(args.output, disp_color)
    
    # Save raw disparity
    raw_output = args.output.replace('.png', '_raw.png')
    disp_16bit = (disp * 256).astype(np.uint16)
    cv2.imwrite(raw_output, disp_16bit, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    print(f"Saving raw disparity to: {raw_output}")
    
    print("=" * 50)
    print("✓ Test completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    test_trt()
