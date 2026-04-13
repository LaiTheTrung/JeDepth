# Export tensorrt engine 

trtexec --onnx=fast_acvnet_generalization_opset16_256x320.onnx --saveEngine=fast_acv_256x320_f32.engine --memPoolSize=workspace:1024 --fp16

trtexec \
  --onnx=fast_acvnet_generalization_opset16_256x320.onnx \
  --shapes=left_image:1x3x256x320,right_image:1x3x256x320 \
  --saveEngine=fast_acv_256x320_f16.trt \
  --fp16

 trtexec   --loadEngine=fast_acv_256x320_f16.engine   --avgRuns=100   --iterations=100   --warmUp=200   --duration=0

## Benchmark Results

| Name | Speed (ms) | Dtype | EPE | SIZE |
|------|------------|-------|-----|------|
| Fast-ACVNet (no GridSample) | 142 | FP32 | - | 288x480|
| Fast-ACVNet (GridSample) | 129 | FP32 | - | 288x480|
| Fast-ACVNet+ (no GridSample) | 152 | FP32 | - | 288x480|
| Fast-ACVNet+ (GridSample) | 144 | FP32 | - | 288x480|
| Fast-ACVNet (no GridSample) | 102 | FP16 | - | 288x480|
| Fast-ACVNet (GridSample) | 94 | FP16 | - | 288x480|
| Fast-ACVNet+ (no GridSample) |  | FP16 | - | 288x480|
| Fast-ACVNet+ (GridSample) | 98 | FP16 | - | 288x480|
| Fast-ACVNet (GridSample) | 75 | FP32 | - | 256x320|
| Fast-ACVNet (GridSample) | 54 | FP16 | - | 256x320|
# Test inference
## ONNX
```bash
python test_onnx.py --model fast_acvnet_plus_generalization_opset16_288x480.onnx \
    --left ./asset/left.png --right ./asset/right.png \
    --output ./out/disparity_onnx.png --maxdisp 192 --use_gpu
```
## Tensorrt
```bash
python test_trt.py \
  --onnx fast_acvnet_generalization_opset16_256x320.onnx \
  --engine fast_acv_256x320_f32.engine \
  --left ./asset/left.png \
  --right ./asset/right.png \
  --output ./out/disparity_trt.png \
  --maxdisp 192 \
  --streams 1
```
