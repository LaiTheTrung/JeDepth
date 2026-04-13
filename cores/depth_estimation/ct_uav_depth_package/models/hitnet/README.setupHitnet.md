# Download pretrained model
cd models
./download_model.sh
# Convert tensorrt
```bash
#chose a model for conversion -> copy it to /models
cd models
sudo systemctl enable nvpmodel
sudo jetson_clocks --fan

trtexec \
  --onnx=hitnet_1x240x320_model_float32.onnx \
  --saveEngine=hitnet_1x240x320_model_float32.engine \
  --memPoolSize=workspace:4096 \
  --fp16 \
  --useCudaGraph \
  --useSpinWait \
  --warmUp=100 \
  --avgRuns=200 \
  --duration=10 \
  --verbose \
  2>&1 | tee trt_build.log

trtexec \
  --onnx=hitnet_1x240x320_model_float32.onnx \
  --saveEngine=hitnet_240x320_fp32_d2slam.engine \
  --memPoolSize=workspace:4096 \
  --useCudaGraph \
  --useSpinWait \
  --warmUp=100 \
  --avgRuns=200 \
  --duration=10 \
  --verbose \
  2>&1 | tee trt_build.log
```
Check the inference speed
```bash
 trtexec   --loadEngine=hitnet_240x320_fp16_opt_level_d2slam.engine   --avgRuns=100   --iterations=300   --warmUp=200   --duration=0
 ```
# Test the ONNX Inference
```bash
python3 test_onnx.py \
  --onnx_path models/hitnet_1x240x320_model_float16_quant_opt.onnx \
  --image_left images/left.jpg \
  --image_right images/right.jpg \
  --gt_depth images/depth_gt.png \
  --width 320 \
  --height 240 \
  --baseline 0.8 \
  --fov_v 140.0 \
  --fov_h 140.0

python3 test_onnx.py \
  --onnx_path models/hitnet_1x240x320_model_float32.onnx \
  --image_left images/left.jpg \
  --image_right images/right.jpg \
  --gt_depth images/depth_gt.png \
  --width 320 \
  --height 240 \
  --baseline 0.8 \
  --fov_v 140.0 \
  --fov_h 140.0
```
# Test the RTR inference
```bash
cd ..
# Test cơ bản
python test_hitnet.py \
    --onnx_path models/hitnet_1x240x320_model_float16_quant_opt.onnx \
    --engine_path models/hitnet_240x320_fp16_opt_level_d2slam.engine \
    --stream_number 1

#Test cơ bản với ground truth:
python test_hitnet.py \
    --onnx_path models/hitnet_1x240x320_model_float16_quant_opt.onnx \
    --engine_path models/hitnet_240x320_fp16_opt_level_d2slam.engine \
    --image_left images/left.jpg \
    --image_right images/right.jpg \
    --stream_number 1 \
    --baseline 0.8 \
    --fov_v 140.0 \
    --fov_h 140.0 \
    --visualize

# Test với calibration parameters:
python test_hitnet.py \
    --onnx_path models/hitnet_1x240x320_model_float16_quant_opt.onnx \
    --engine_path models/hitnet_240x320_fp16_opt_level_d2slam.engine \
    --image_left images/left.jpg \
    --image_right images/right.jpg \
    --gt_depth images/depth_gt.png \
    --baseline 0.8 \
    --fov_v 140.0 \
    --fov_h 140.0 \
    --visualize \
    --stream_number 2
    --save_metrics results.json 
  
```