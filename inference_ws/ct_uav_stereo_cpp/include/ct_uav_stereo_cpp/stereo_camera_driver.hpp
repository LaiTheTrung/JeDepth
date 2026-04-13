#pragma once

// Driver for the stereo camera using libargus library.
// Author: Lai The Trung
#include <iostream>
#include <iomanip>
#include <csignal>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>

#include "ArgusHelpers.h"
#include "CUDAHelper.h"
#include "CommonOptions.h"
#include "EGLGlobal.h"
#include "Error.h"
#include "Thread.h"
#include <cuda.h>
#include <cudaEGL.h>


#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "PreviewConsumer.h"
#include "UniquePointer.h"


#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <string.h>
#include <map>


#include <Argus/Ext/BlockingSessionCameraProvider.h>
#include <ct_uav_stereo_cpp/cu_color_cvt.h>
#include <ct_uav_stereo_cpp/histogram.h>

namespace CtUAVStereoCpp
{
using namespace Argus;
using namespace EGLStream;
// //====================================================================================
// //===============================METADATA INITIALIZE==================================
// //====================================================================================
// Constants.
#define PRODUCER_PRINT(...) printf("PRODUCER: " __VA_ARGS__)
#define CONSUMER_PRINT(...) printf("CONSUMER: " __VA_ARGS__)
#define APP_PRINT(...) printf("SYNC STEREO RAW INJ APP: " __VA_ARGS__)
#define PRODUCER_PRINT(...) printf("PRODUCER: " __VA_ARGS__)

#define MAX_MODULE_STRING 16
#define MAX_CAM_DEVICE 2
#define left_device 0
#define right_device 1
#define USING_GPU
#define ASSEMBLE_IMAGE

#define SENSOR_MODE_INDEX 3 // 0: 3280x2464@21fps, 1: 3280x1848@28fps, 2: 1920x1080@30fps, 2: 1640x1232@30fps, 3: 1280x720@60fps
static Size2D<uint32_t> STREAM_SIZE(1640, 1232);

/// Camera Config
static const uint32_t    DEFAULT_CAPTURE_TIME  = 20; // In seconds.
static const uint64_t    DEFAULT_FRAME_DURATION = 33000000U; //30FPS
static const uint64_t    DEFAULT_FRAME_RATE = 30; //30FPS

static const Range<float> ISP_DIGITAL_GAIN_RANGE(1, 8); // should't set too high, it cause latency. don't want to use isp digital gain range cuz it make noise
static const Range<uint64_t> EXPOSURE_TIME_RANGE(1700000, 15000000); //nano second maximum = DEFAULT_FRAME_DURATION
static const uint64_t DEFAULT_EXPOSURE_TIME = 8000000; //nano second maximum = DEFAULT_FRAME_DURATION
static const Range<float> GAIN_RANGE(1.0, 10.625); //infact (1,10.625)
static const float DEFAULT_GAIN = 2.0; //nano second maximum = DEFAULT_FRAME_DURATION


static const bool AE_LOCK_ENABLE = true;
static const bool AE_LOCK_DISABLE = false;
static const float TARGET_EXPOSURE_LEVEL = 0.5475;
static const float TARGET_EXPOSURE_LEVEL_RANGE_TOLERANCE = 0.004;
const Range<float> TARGET_RANGE(TARGET_EXPOSURE_LEVEL - TARGET_EXPOSURE_LEVEL_RANGE_TOLERANCE,
                                   TARGET_EXPOSURE_LEVEL + TARGET_EXPOSURE_LEVEL_RANGE_TOLERANCE);
/// maximum number of RAW captures to do, default is to save all the frames as long
/// as the preview is running
static const uint32_t    MAX_NUM_RAW_CAPTURES = 0xFFFFFFFF;
static const uint32_t    EGL_STREAM_BUFFERS = 2;
static const bool        DEFAULT_NVRAW_CAPTURE = false;

class SyncStereoConsumerThread;
class UserAutoExposureTeardown;
class StereoDriverNode;

typedef struct
{
    char moduleName[MAX_MODULE_STRING];
    int camDevice[MAX_CAM_DEVICE];
    ICameraProperties *iCameraProperties[MAX_CAM_DEVICE];
    Argus::SensorMode *sensorMode[MAX_CAM_DEVICE];
    ISensorMode *iSensorMode[MAX_CAM_DEVICE];
    UniqueObj<OutputStream> stream[MAX_CAM_DEVICE];
    UniqueObj<InputStream> inStream[MAX_CAM_DEVICE];
    UniqueObj<CaptureSession> captureSession[MAX_CAM_DEVICE];
    UniqueObj<OutputStreamSettings> streamSettings[MAX_CAM_DEVICE];
    UniqueObj<Request> request[MAX_CAM_DEVICE];
    ICaptureSession* iCaptureSession[MAX_CAM_DEVICE];
    SyncStereoConsumerThread *stereoYuvConsumer;
    int sensorCount;
    bool initialized;
    bool using_gpu;
    bool assembled_mode;
    int conversion_type;
    bool auto_expose;
} ModuleInfo;

static uint8_t* oBuffer= new uint8_t[STREAM_SIZE.width() * STREAM_SIZE.height() * 3];


//====================================================================================
//===============================ARGUS CONSUMER========================================
//====================================================================================

class SyncStereoConsumerThread : public ArgusSamples::Thread
{
public:
    explicit SyncStereoConsumerThread(
        IEGLOutputStreamSettings *iEGLStreamSettings,
        ModuleInfo *modInfo,
        OutputStream *stream
    ):m_cudaContext(0),
    m_cuStreamLeft(NULL),
    m_cuStreamRight(NULL),
    using_gpu(false)
    {
        using_gpu = modInfo->using_gpu;
        conversion_type = modInfo->conversion_type;
        auto_expose = modInfo->auto_expose;
        assembled_mode = modInfo->assembled_mode;

        CONSUMER_PRINT("Start using GPU: %s\n", using_gpu ? "true" : "false");
        if (using_gpu){
        cu_leftStream = interface_cast<IEGLOutputStream>(modInfo->stream[left_device]);
        cu_rightStream = interface_cast<IEGLOutputStream>(modInfo->stream[right_device]);
        }

        else {
            m_leftStream = modInfo->stream[left_device].get();
            m_rightStream = modInfo->stream[right_device].get();
        }
        strcpy(m_moduleName,modInfo->moduleName);
    }
    ~SyncStereoConsumerThread()
    {
        CONSUMER_PRINT("DESTRUCTOR  ... \n");
    }

    float frameGain = DEFAULT_GAIN;
    uint64_t frameExposureTime = DEFAULT_EXPOSURE_TIME;
    float ispGain = ISP_DIGITAL_GAIN_RANGE.min();
    void get_frames(cv::Mat& left, cv::Mat& right, double& latency) const;
    void get_frames(cv::Mat& concat, double& latency) const;
    void get_image_msg (sensor_msgs::msg::Image& msg, double& latency) const;
    float get_frame_gain() const { return frameGain; }
    float get_isp_gain() const { return ispGain; }
    uint64_t get_frame_exposure_time() const { return frameExposureTime; }
    
private:
    /** @name Thread methods */
    /**@{*/
    virtual bool threadInitialize();
    virtual bool threadExecute();
    virtual bool threadShutdown();
    /**@}*/

    /* Assumption: We only have a Left-Right pair.
     * OutputStream and FrameConsumer should be created to a vector of
     * MAX_CAM_DEVICE size.
     */
    bool using_gpu;
    bool assembled_mode = false;
    bool compressed_mode = false;
    bool auto_expose = false;
    int conversion_type = cvt_YUV2GRAY; // Default: GRAY
    CameraDevice* m_cameraDevice;
    IEGLOutputStreamSettings* m_iEGLStreamSettings;
    OutputStream *m_leftStream;
    OutputStream *m_rightStream;
    char m_moduleName[MAX_MODULE_STRING];
    UniqueObj<FrameConsumer> m_leftConsumer;
    UniqueObj<FrameConsumer> m_rightConsumer;

    double capture_moment_ms_ = 0.0f; // Time when the current frame was captured, in milliseconds since epoch.
    Argus::Size2D<uint32_t> m_bayerSize;  // Size of Bayer input.
    Argus::Size2D<uint32_t> m_outputSize; // Size of RGBA output.
    size_t image_size =  STREAM_SIZE.width() * STREAM_SIZE.height();

    CUresult cuResult_left;
    CUresult cuResult_right;
    IEGLOutputStream *cu_leftStream;
    IEGLOutputStream *cu_rightStream;
    CUeglStreamConnection m_cuStreamLeft; // CUDA handle to Bayer stream.
    CUeglStreamConnection m_cuStreamRight; // CUDA handle to Bayer stream.
    CUgraphicsResource m_resource;
    CUcontext m_cudaContext;

    cv::Mat leftFrame;
    cv::Mat rightFrame;
    cv::Mat concatFrame;
    mutable std::mutex frame_mutex_;
    void auto_exposure_control(float curExposureLevel);
};

//====================================================================================
//============================POST-PROCESSING========================================
//====================================================================================

class CudaFrameAcquire {
  public:
    CudaFrameAcquire(CUeglStreamConnection& connection);
    ~CudaFrameAcquire();
    bool generateHistogram(unsigned int histogramData[HISTOGRAM_BINS], uint64_t image_size, float *curExposureLevel);
        bool convert2CPU(bool leftFrame, int conversionType);
    bool hasValidFrame() const;
        const uint8_t* getOutputBuffer() const;
        size_t getFrameSize() const;
  private:
    bool checkFormat() const;
    CUeglStreamConnection& m_connection;
    CUgraphicsResource m_resource;
    CUeglFrame m_frame;
    CUstream m_stream;
        std::vector<uint8_t> m_outputBuffer;
        size_t m_frameSize = 0;
};
}