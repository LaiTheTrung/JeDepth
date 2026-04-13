#include <gst/gst.h>
#include <nvbufsurface.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>

#include <jetson-utils/videoOutput.h>
#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/logging.h>

#include <vector>
#include <iostream>
#include <chrono>
#include <unordered_map>

// --- Global state ---
videoOutput* output = nullptr;
GMainLoop* loop = nullptr;
GstElement* pipeline = nullptr;
int sensor_ids[4] = {0, 1, 2, 3};

uchar *unified_final_buffer = nullptr;
cv::cuda::GpuMat gpu_final_frame;
cudaStream_t copy_streams[4];  // One stream per sensor

auto loop_start = std::chrono::high_resolution_clock::now();

// Map for persistent EGL to CUgraphicsResource
std::unordered_map<EGLImageKHR, CUgraphicsResource> egl_to_cu_resource;

void InitCudaContextOnce()
{
    // Create a thread-local flag to ensure we only init once per thread
    thread_local bool initialized = false;
    if (!initialized) {
        cudaFree(0);  // Forces primary context setup
        initialized = true;
    }
}

void CvCudaProcessEGLImage(EGLImageKHR egl_image, unsigned int pitch, int sensorID)
{
    using namespace std::chrono;

    auto t_start = high_resolution_clock::now();

    CUresult status;
    CUeglFrame eglFrame;
    CUgraphicsResource cu_res = nullptr;

    auto t_reg_start = high_resolution_clock::now();
    // Register only once
    auto it = egl_to_cu_resource.find(egl_image);
    if (it == egl_to_cu_resource.end()) {

        if (egl_image == EGL_NO_IMAGE_KHR || egl_image == nullptr) {
            g_printerr("Invalid EGLImage\n");
            return;
        }

        status = cuGraphicsEGLRegisterImage(&cu_res, egl_image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
        if (status != CUDA_SUCCESS) {
            const char* err_str = nullptr;
            cuGetErrorString(status, &err_str);
            g_printerr("cuGraphicsEGLRegisterImage failed: %s\n", err_str);
            return;
        }
        egl_to_cu_resource[egl_image] = cu_res;
    } else {
        cu_res = it->second;
    }
    auto t_reg_end = high_resolution_clock::now();

    auto t_map_start = high_resolution_clock::now();
    status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, cu_res, 0, 0);
    if (status != CUDA_SUCCESS) {
        g_printerr("cuGraphicsResourceGetMappedEglFrame failed\n");
        return;
    }
    status = cuCtxSynchronize();  // Can be optimized later
    auto t_map_end = high_resolution_clock::now();

    auto t_copy_start = high_resolution_clock::now();
    cv::cuda::GpuMat d_mat(eglFrame.height, pitch / 4, CV_8UC4, (uchar*) eglFrame.frame.pPitch[0]);

    const int w = 512, h = 320;
    if (sensorID == 0) d_mat.copyTo(gpu_final_frame(cv::Rect(0, 0, w, h)));
    else if (sensorID == 1) d_mat.copyTo(gpu_final_frame(cv::Rect(w, 0, w, h)));
    else if (sensorID == 2) d_mat.copyTo(gpu_final_frame(cv::Rect(0, h, w, h)));
    else if (sensorID == 3) d_mat.copyTo(gpu_final_frame(cv::Rect(w, h, w, h)));
    auto t_copy_end = high_resolution_clock::now();

    auto t_render_start = high_resolution_clock::now();
    if (sensorID == 0 && output) {
        output->Render((uchar4*)unified_final_buffer, 1024, 640);
        auto end_time = high_resolution_clock::now();
        float fps = 1000.0f / duration_cast<milliseconds>(end_time - loop_start).count();
        //std::cout << "FPS: " << fps << std::endl;
        loop_start = end_time;
    }
    auto t_render_end = high_resolution_clock::now();

    // Total time
    auto t_end = high_resolution_clock::now();

    // Print profile results only for sensor 0 (or all if you prefer)
    /*if (sensorID == 0) {
        auto dur = [](auto a, auto b) { return duration_cast<microseconds>(b - a).count(); };

        std::cout << "[Profile] Sensor " << sensorID
                  << " | Register: " << dur(t_reg_start, t_reg_end) << "us"
                  << " | Map+Sync: " << dur(t_map_start, t_map_end) << "us"
                  << " | Copy: "     << dur(t_copy_start, t_copy_end) << "us"
                  << " | Render: "   << dur(t_render_start, t_render_end) << "us"
                  << " | Total: "    << dur(t_start, t_end) << "us"
                  << std::endl;
    }*/
}

static GstPadProbeReturn conv_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
    using namespace std::chrono;
    auto t_probe_start = high_resolution_clock::now();

    InitCudaContextOnce();  // ensures once-per-thread context init

    int sensorID = *(int*)u_data;
    GstBuffer *buffer = (GstBuffer *)info->data;
    GstMapInfo map = {0};

    auto t_map_start = high_resolution_clock::now();
    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        g_printerr("Failed to map GstBuffer\n");
        return GST_PAD_PROBE_OK;
    }
    auto t_map_end = high_resolution_clock::now();

    NvBufSurface* surf = (NvBufSurface*)map.data;
    NvBufSurfaceParams& params = surf->surfaceList[0];

    auto t_eglmap_start = high_resolution_clock::now();
    if (NvBufSurfaceMapEglImage(surf, 0) != 0) {
        g_printerr("Failed to map surface to EGLImage\n");
        gst_buffer_unmap(buffer, &map);
        return GST_PAD_PROBE_OK;
    }
    auto t_eglmap_end = high_resolution_clock::now();

    EGLImageKHR egl_image = params.mappedAddr.eglImage;
    if (!egl_image) {
        g_printerr("No EGLImage mapped from surface\n");
        gst_buffer_unmap(buffer, &map);
        return GST_PAD_PROBE_OK;
    }

    auto t_cuda_start = high_resolution_clock::now();
    CvCudaProcessEGLImage(egl_image, params.pitch, sensorID);
    auto t_cuda_end = high_resolution_clock::now();

    auto t_eglunmap_start = high_resolution_clock::now();
    NvBufSurfaceUnMapEglImage(surf, 0);
    auto t_eglunmap_end = high_resolution_clock::now();

    gst_buffer_unmap(buffer, &map);
    auto t_probe_end = high_resolution_clock::now();

    /*if (sensorID == 0) {
        auto us = [](auto a, auto b) { return duration_cast<microseconds>(b - a).count(); };

        std::cout << "[Probe Profile] Sensor " << sensorID
                  << " | gst_map: "     << us(t_map_start, t_map_end) << "us"
                  << " | EGLMap: "      << us(t_eglmap_start, t_eglmap_end) << "us"
                  << " | CUDA proc: "   << us(t_cuda_start, t_cuda_end) << "us"
                  << " | EGLUnmap: "    << us(t_eglunmap_start, t_eglunmap_end) << "us"
                  << " | Total Probe: " << us(t_probe_start, t_probe_end) << "us"
                  << std::endl;
    }*/

    return GST_PAD_PROBE_OK;
}

bool run_capture()
{
    InitCudaContextOnce();
    gst_init(NULL, NULL);
    loop = g_main_loop_new(NULL, FALSE);

    std::string pipeline_str;
    for (int i = 0; i < 4; ++i) {
        pipeline_str += "nvarguscamerasrc sensor-id=" + std::to_string(i) +
                        " ! video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1" +
                        " ! nvvidconv name=conv" + std::to_string(i) +
                        " ! video/x-raw(memory:NVMM),width=512,height=320,format=RGBA" +
                        " ! appsink max-buffers=1 drop=true emit-signals=true sync=false ";
    }

    pipeline = gst_parse_launch(pipeline_str.c_str(), nullptr);
    if (!pipeline) {
        g_printerr("Failed to create pipeline\n");
        return false;
    }

    for (int i = 0; i < 4; ++i) {
        GstElement* conv = gst_bin_get_by_name(GST_BIN(pipeline), ("conv" + std::to_string(i)).c_str());
        GstPad* pad = gst_element_get_static_pad(conv, "src");
        gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, conv_src_pad_buffer_probe, &sensor_ids[i], nullptr);
        gst_object_unref(pad);
        gst_object_unref(conv);
    }

    size_t final_frame_size = 1024 * 640 * 4;
    cudaError_t err = cudaHostAlloc((void**)&unified_final_buffer, final_frame_size, cudaHostAllocMapped);
    if (err != cudaSuccess)
        err = cudaMallocManaged(&unified_final_buffer, final_frame_size);

    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate final frame buffer: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    gpu_final_frame = cv::cuda::GpuMat(640, 1024, CV_8UC4, unified_final_buffer);
    std::cout << "gpu_final_frame allocated\n";

    for (int i = 0; i < 4; ++i) {
        cudaStreamCreate(&copy_streams[i]);
    }

    output = videoOutput::Create("webrtc://@:8554/output");
    if (!output) {
        LogError("failed to create output stream\n");
        cudaFree(unified_final_buffer);
        return false;
    }

    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (gst_element_get_state(pipeline, nullptr, nullptr, -1) == GST_STATE_CHANGE_FAILURE) {
        g_error("Failed to go into PLAYING state");
    }

    g_main_loop_run(loop);

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    for (auto& kv : egl_to_cu_resource) {
        cuGraphicsUnregisterResource(kv.second);
    }
    egl_to_cu_resource.clear();

    cudaFree(unified_final_buffer);
    SAFE_DELETE(output);

    return true;
}

int main(int argc, char* argv[])
{
    return run_capture() ? 0 : 1;
}