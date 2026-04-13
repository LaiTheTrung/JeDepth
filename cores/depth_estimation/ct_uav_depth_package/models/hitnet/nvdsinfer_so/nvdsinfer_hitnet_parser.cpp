/**
 * Copyright (c) 2024, Custom Implementation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
 * @file nvdsinfer_hitnet_parser.cpp
 * @brief Custom parser for HitNet stereo depth estimation model in DeepStream
 * 
 * This parser processes the output from HitNet model which produces disparity maps
 * for stereo depth estimation. It converts the raw model output into a format
 * suitable for further processing in DeepStream pipeline.
 */

#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "nvdsinfer_custom_impl.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/**
 * @brief Parse HitNet disparity output
 * 
 * HitNet model outputs a disparity map with dimensions [batch, 1, height, width]
 * This function extracts the disparity values and stores them for downstream processing
 * 
 * @param outputLayersInfo Array of output layer information
 * @param numOutputLayers Number of output layers
 * @param width Output width
 * @param height Output height
 * @param objectList Output object list (not used for disparity, kept for compatibility)
 * @return true if parsing successful, false otherwise
 */
extern "C" bool NvDsInferParseCustomHitnet(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    if (outputLayersInfo.empty()) {
        std::cerr << "ERROR: No output layers found" << std::endl;
        return false;
    }

    // HitNet typically has one output layer: disparity map
    const NvDsInferLayerInfo &disparityLayer = outputLayersInfo[0];
    
    // Expected output format: [batch, channels, height, width]
    if (disparityLayer.inferDims.numDims != 4 && disparityLayer.inferDims.numDims != 3) {
        std::cerr << "ERROR: Unexpected output dimensions: " 
                  << disparityLayer.inferDims.numDims << std::endl;
        return false;
    }

    // Extract dimensions
    int batch_size = (disparityLayer.inferDims.numDims == 4) ? 
                     disparityLayer.inferDims.d[0] : 1;
    int channels = (disparityLayer.inferDims.numDims == 4) ? 
                   disparityLayer.inferDims.d[1] : disparityLayer.inferDims.d[0];
    int height = (disparityLayer.inferDims.numDims == 4) ? 
                 disparityLayer.inferDims.d[2] : disparityLayer.inferDims.d[1];
    int width = (disparityLayer.inferDims.numDims == 4) ? 
                disparityLayer.inferDims.d[3] : disparityLayer.inferDims.d[2];

    std::cout << "HitNet Parser: Processing disparity map" << std::endl;
    std::cout << "  Batch: " << batch_size << std::endl;
    std::cout << "  Channels: " << channels << std::endl;
    std::cout << "  Height: " << height << std::endl;
    std::cout << "  Width: " << width << std::endl;

    // Get pointer to output data
    float* disparity_data = (float*)disparityLayer.buffer;
    
    if (!disparity_data) {
        std::cerr << "ERROR: Output buffer is null" << std::endl;
        return false;
    }

    // Calculate statistics for debugging
    float min_disp = std::numeric_limits<float>::max();
    float max_disp = std::numeric_limits<float>::min();
    float sum_disp = 0.0f;
    int valid_pixels = 0;
    
    int total_pixels = height * width;
    
    for (int i = 0; i < total_pixels; i++) {
        float disp = disparity_data[i];
        
        if (std::isfinite(disp) && disp > 0) {
            min_disp = MIN(min_disp, disp);
            max_disp = MAX(max_disp, disp);
            sum_disp += disp;
            valid_pixels++;
        }
    }

    if (valid_pixels > 0) {
        float mean_disp = sum_disp / valid_pixels;
        std::cout << "  Disparity stats:" << std::endl;
        std::cout << "    Min: " << min_disp << std::endl;
        std::cout << "    Max: " << max_disp << std::endl;
        std::cout << "    Mean: " << mean_disp << std::endl;
        std::cout << "    Valid pixels: " << valid_pixels << "/" << total_pixels 
                  << " (" << (100.0f * valid_pixels / total_pixels) << "%)" << std::endl;
    } else {
        std::cerr << "WARNING: No valid disparity values found" << std::endl;
    }

    // Store disparity map in user metadata for downstream processing
    // Note: In a real implementation, you would attach this to frame metadata
    // For now, we just validate the parsing was successful
    
    return true;
}

/**
 * @brief Parse HitNet output with depth conversion
 * 
 * This version converts disparity to depth using stereo geometry:
 * depth = (baseline * focal_length) / disparity
 * 
 * @param outputLayersInfo Array of output layer information
 * @param numOutputLayers Number of output layers
 * @param baseline Stereo baseline distance in meters
 * @param focal_length Focal length in pixels
 * @param objectList Output object list (not used)
 * @return true if parsing successful, false otherwise
 */
extern "C" bool NvDsInferParseCustomHitnetWithDepth(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    // Get disparity output
    if (outputLayersInfo.empty()) {
        std::cerr << "ERROR: No output layers found" << std::endl;
        return false;
    }

    const NvDsInferLayerInfo &disparityLayer = outputLayersInfo[0];
    
    // Extract dimensions
    int height = disparityLayer.inferDims.d[2];
    int width = disparityLayer.inferDims.d[3];
    
    float* disparity_data = (float*)disparityLayer.buffer;
    
    if (!disparity_data) {
        std::cerr << "ERROR: Output buffer is null" << std::endl;
        return false;
    }

    // Stereo parameters (these should be configurable via config file)
    const float baseline = 0.8f;  // 80cm baseline (example)
    const float focal_length = 320.0f;  // pixels (example)
    const float min_disparity = 1.0f;  // Minimum valid disparity
    
    std::cout << "HitNet Parser with Depth Conversion" << std::endl;
    std::cout << "  Baseline: " << baseline << " m" << std::endl;
    std::cout << "  Focal length: " << focal_length << " px" << std::endl;

    // Calculate depth statistics
    float min_depth = std::numeric_limits<float>::max();
    float max_depth = std::numeric_limits<float>::min();
    float sum_depth = 0.0f;
    int valid_pixels = 0;
    
    int total_pixels = height * width;
    
    // Convert disparity to depth
    std::vector<float> depth_map(total_pixels);
    
    for (int i = 0; i < total_pixels; i++) {
        float disp = disparity_data[i];
        
        if (disp > min_disparity) {
            // Depth = (baseline * focal_length) / disparity
            float depth = (baseline * focal_length) / disp;
            depth_map[i] = depth;
            
            if (std::isfinite(depth) && depth > 0) {
                min_depth = MIN(min_depth, depth);
                max_depth = MAX(max_depth, depth);
                sum_depth += depth;
                valid_pixels++;
            }
        } else {
            depth_map[i] = 0.0f;  // Invalid depth
        }
    }

    if (valid_pixels > 0) {
        float mean_depth = sum_depth / valid_pixels;
        std::cout << "  Depth stats:" << std::endl;
        std::cout << "    Min: " << min_depth << " m" << std::endl;
        std::cout << "    Max: " << max_depth << " m" << std::endl;
        std::cout << "    Mean: " << mean_depth << " m" << std::endl;
        std::cout << "    Valid pixels: " << valid_pixels << "/" << total_pixels 
                  << " (" << (100.0f * valid_pixels / total_pixels) << "%)" << std::endl;
    }

    return true;
}

/**
 * @brief Check if custom parser is compatible with model
 * 
 * @param modelName Name of the model
 * @param numOutputLayers Number of output layers
 * @return true if compatible, false otherwise
 */
extern "C" bool NvDsInferParseCustomHitnetCheck(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo)
{
    if (outputLayersInfo.empty()) {
        return false;
    }

    // Check if output layer format matches HitNet
    const NvDsInferLayerInfo &layer = outputLayersInfo[0];
    
    // HitNet outputs [batch, 1, height, width]
    if (layer.inferDims.numDims == 4) {
        int channels = layer.inferDims.d[1];
        if (channels == 1) {
            std::cout << "HitNet parser: Model compatible" << std::endl;
            return true;
        }
    } else if (layer.inferDims.numDims == 3) {
        // Some models might output [1, height, width]
        std::cout << "HitNet parser: Model compatible (3D output)" << std::endl;
        return true;
    }

    std::cerr << "HitNet parser: Model NOT compatible" << std::endl;
    return false;
}

/**
 * @brief Create 3D point cloud from disparity map
 * 
 * This function demonstrates how to convert disparity to 3D points
 * In a real implementation, this would be called from a separate plugin
 * 
 * @param disparity Disparity map
 * @param width Image width
 * @param height Image height
 * @param baseline Stereo baseline in meters
 * @param focal_length Focal length in pixels
 * @param cx Principal point x
 * @param cy Principal point y
 * @param points Output 3D points
 */
void disparityToPointCloud(
    const float* disparity,
    int width, int height,
    float baseline, float focal_length,
    float cx, float cy,
    std::vector<float>& points)
{
    points.clear();
    points.reserve(width * height * 3);  // x, y, z for each point
    
    const float min_disparity = 1.0f;
    const float max_depth = 50.0f;  // Maximum valid depth in meters
    
    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            int idx = v * width + u;
            float disp = disparity[idx];
            
            if (disp > min_disparity) {
                // Calculate depth
                float z = (baseline * focal_length) / disp;
                
                if (z > 0 && z < max_depth) {
                    // Calculate 3D coordinates
                    float x = (u - cx) * z / focal_length;
                    float y = (v - cy) * z / focal_length;
                    
                    points.push_back(x);
                    points.push_back(y);
                    points.push_back(z);
                }
            }
        }
    }
}

/**
 * @brief Save disparity map to file for debugging
 * 
 * @param disparity Disparity map
 * @param width Image width
 * @param height Image height
 * @param filename Output filename
 */
void saveDisparityMap(const float* disparity, int width, int height, 
                     const char* filename)
{
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        std::cerr << "ERROR: Cannot open file for writing: " << filename << std::endl;
        return;
    }
    
    // Save as binary float
    fwrite(&width, sizeof(int), 1, fp);
    fwrite(&height, sizeof(int), 1, fp);
    fwrite(disparity, sizeof(float), width * height, fp);
    
    fclose(fp);
    std::cout << "Saved disparity map to: " << filename << std::endl;
}

// Registry for custom parsers
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomHitnet);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomHitnetWithDepth);