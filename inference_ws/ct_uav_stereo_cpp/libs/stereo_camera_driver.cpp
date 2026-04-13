#include "ct_uav_stereo_cpp/stereo_camera_driver.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cuda.h>
#include <cudaEGL.h>
#include "CUDAHelper.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cinttypes>



namespace CtUAVStereoCpp
{
using namespace std;
using namespace Argus;
using namespace EGLStream;
using namespace ArgusSamples;

bool SyncStereoConsumerThread::threadInitialize(){
    if (conversion_type == cvt_YUV2GRAY) {
        leftFrame = cv::Mat(STREAM_SIZE.height(), STREAM_SIZE.width(), CV_8UC1);
        rightFrame = cv::Mat(STREAM_SIZE.height(), STREAM_SIZE.width(), CV_8UC1);
        concatFrame = cv::Mat(STREAM_SIZE.height(), STREAM_SIZE.width() * 2, CV_8UC1);
    }
    else {
        leftFrame = cv::Mat(STREAM_SIZE.height(), STREAM_SIZE.width(), CV_8UC3);
        rightFrame = cv::Mat(STREAM_SIZE.height(), STREAM_SIZE.width(), CV_8UC3);
        concatFrame = cv::Mat(STREAM_SIZE.height(), STREAM_SIZE.width() * 2, CV_8UC3);
    }
    PROPAGATE_ERROR(initCUDA(&m_cudaContext));

    CONSUMER_PRINT("Connecting CUDA consumer to left stream\n");

    CUresult cuResult = cuEGLStreamConsumerConnect(&m_cuStreamLeft, cu_leftStream->getEGLStream());
    if (cuResult != CUDA_SUCCESS) ORIGINATE_ERROR("Unable to connect CUDA to EGLStream (%s)", getCudaErrorString(cuResult));
    
    cuResult = cuEGLStreamConsumerConnect(&m_cuStreamRight, cu_rightStream->getEGLStream());
    if (cuResult != CUDA_SUCCESS) ORIGINATE_ERROR("Unable to connect CUDA to EGLStream (%s)", getCudaErrorString(cuResult));
    return true;
}

bool SyncStereoConsumerThread::threadShutdown(){
            // Disconnect from the streams.
    if (using_gpu){
    cuEGLStreamConsumerDisconnect(&m_cuStreamLeft);
    cuEGLStreamConsumerDisconnect(&m_cuStreamRight);

    PROPAGATE_ERROR(cleanupCUDA(&m_cudaContext));
    cu_leftStream->disconnect();
    cu_rightStream->disconnect();
    }
    CONSUMER_PRINT("Done.\n");
    return true;
}

void SyncStereoConsumerThread::auto_exposure_control(float curExposureLevel) {
    if (curExposureLevel >= TARGET_RANGE.min() && curExposureLevel <= TARGET_RANGE.max()) {
        return;
    }

    float error = TARGET_EXPOSURE_LEVEL - curExposureLevel;

    // --- Exposure Time: dùng additive step thay vì multiplicative ---
    // Tính step tuyệt đối dựa trên range, không dựa trên giá trị hiện tại
    uint64_t exposureRange = EXPOSURE_TIME_RANGE.max() - EXPOSURE_TIME_RANGE.min();
    double exposureStep = 0.1 * error * static_cast<double>(exposureRange);
    double newExposure = static_cast<double>(frameExposureTime) + exposureStep;
    newExposure = std::max(static_cast<double>(EXPOSURE_TIME_RANGE.min()),
                           std::min(newExposure, static_cast<double>(EXPOSURE_TIME_RANGE.max())));
    frameExposureTime = static_cast<uint64_t>(newExposure);

    // --- Chỉ điều chỉnh gain khi exposure đã chạm limit ---
    bool exposureAtLimit = (frameExposureTime <= EXPOSURE_TIME_RANGE.min() && error < 0) ||
                           (frameExposureTime >= EXPOSURE_TIME_RANGE.max() && error > 0);

    if (exposureAtLimit) {
        float gainRange = GAIN_RANGE.max() - GAIN_RANGE.min();
        float gainStep = 0.05f * error * gainRange;
        frameGain = std::clamp(frameGain + gainStep, GAIN_RANGE.min(), GAIN_RANGE.max());

        // Chỉ dùng ISP gain khi analog gain cũng chạm limit
        bool gainAtLimit = (frameGain <= GAIN_RANGE.min() && error < 0) ||
                           (frameGain >= GAIN_RANGE.max() && error > 0);
        if (gainAtLimit) {
            float ispRange = ISP_DIGITAL_GAIN_RANGE.max() - ISP_DIGITAL_GAIN_RANGE.min();
            float ispStep = 0.01f * error * ispRange;
            ispGain = std::clamp(ispGain + ispStep,
                                 ISP_DIGITAL_GAIN_RANGE.min(), ISP_DIGITAL_GAIN_RANGE.max());
        }
    } else {
        // Khi exposure chưa chạm limit → kéo gain về mức thấp nhất (giảm noise)
        float gainDecay = 0.05f;
        if (frameGain > DEFAULT_GAIN) {
            frameGain = std::max(DEFAULT_GAIN, frameGain - gainDecay * frameGain);
        }
        if (ispGain > ISP_DIGITAL_GAIN_RANGE.min()) {
            ispGain = std::max(ISP_DIGITAL_GAIN_RANGE.min(), ispGain - gainDecay * ispGain);
        }
    }
}

inline void copyBufferToMat(
    const uint8_t* buffer,
    cv::Mat& img,
    bool isconcat = false, bool to_left = true)
{
    if (!isconcat){
        std::memcpy(img.data, buffer, img.total() * img.elemSize());
    }
    else {
        int offset = to_left ? 0 : img.cols * img.elemSize() / 2;
        for (int i = 0; i < img.rows; ++i) {
            std::memcpy(img.ptr(i) + offset, buffer + i * img.cols * img.elemSize() / 2, img.cols * img.elemSize() / 2);
        }
    }

}

bool SyncStereoConsumerThread::threadExecute(){
    if (!cu_leftStream || !cu_rightStream) {
        CONSUMER_PRINT("Invalid EGL streams (left/right null).\n");
        return false;
    }
    CONSUMER_PRINT("Waiting for stream connection...\n");
    cu_leftStream->waitUntilConnected();
    cu_rightStream->waitUntilConnected();
    unsigned int histogramLeft[HISTOGRAM_BINS];
    while (rclcpp::ok()){
        EGLint streamState = EGL_STREAM_STATE_CONNECTING_KHR;
        if (!eglQueryStreamKHR(
            cu_leftStream->getEGLDisplay(),
            cu_leftStream->getEGLStream(),
            EGL_STREAM_STATE_KHR,
            &streamState) || (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR))
        {
            CONSUMER_PRINT("left : EGL stream query failed or disconnected (state=0x%x)\n", streamState);
            break;
        }
        if (!eglQueryStreamKHR(
                cu_rightStream->getEGLDisplay(),
                cu_rightStream->getEGLStream(),
                EGL_STREAM_STATE_KHR,
                &streamState) || (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR))
        {
            CONSUMER_PRINT("right : EGL stream query failed or disconnected (state=0x%x)\n", streamState);
            break;
        }
        CudaFrameAcquire right(m_cuStreamRight);
        CudaFrameAcquire left(m_cuStreamLeft);
        auto start = std::chrono::high_resolution_clock::now();
        capture_moment_ms_ = std::chrono::duration<double, std::milli>(start.time_since_epoch()).count();
        if (auto_expose){
            float curExposureLevel = 0.0f;
            if (left.generateHistogram(histogramLeft,(uint64_t)image_size, &curExposureLevel))
                auto_exposure_control(curExposureLevel);
        }
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            PROPAGATE_ERROR(left.convert2CPU(true, conversion_type));
            if (assembled_mode) copyBufferToMat(oBuffer, concatFrame, true, true);
            else copyBufferToMat(oBuffer, leftFrame);
            std::memcpy(leftFrame.data, oBuffer, leftFrame.total() * leftFrame.elemSize());
            PROPAGATE_ERROR(right.convert2CPU(false, conversion_type));
            if (assembled_mode) copyBufferToMat(oBuffer, concatFrame, true, false);
            else copyBufferToMat(oBuffer, rightFrame);
        }
        // auto end = std::chrono::high_resolution_clock::now();
        
        // std::chrono::duration<double, std::milli> duration = end - start;
        // std::cout << "[TIME-DEBUG] Time taken for code execution in one iteration: " << duration.count() << " milliseconds." << std::endl;
        // printf("\n");
    }

    PROPAGATE_ERROR(this->requestShutdown());
    return true;
}

void SyncStereoConsumerThread::get_frames(cv::Mat& leftFrame, cv::Mat& rightFrame, double& latency) const {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (!assembled_mode) {
        leftFrame = this->leftFrame.clone();
        rightFrame = this->rightFrame.clone();
    }
    else {
        leftFrame = this->concatFrame.colRange(0, concatFrame.cols / 2).clone();
        rightFrame = this->concatFrame.colRange(concatFrame.cols / 2, concatFrame.cols).clone();
    }
    auto end = std::chrono::high_resolution_clock::now();
    latency = std::chrono::duration<double, std::milli>(end.time_since_epoch()).count() - capture_moment_ms_;
}

void SyncStereoConsumerThread::get_frames(cv::Mat& frame, double& latency) const {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (assembled_mode) {
        frame = this->concatFrame.clone();
    }
    else {
        // If not in assembled mode, concatenate left and right frames
        cv::hconcat(this->leftFrame, this->rightFrame, concatFrame);
    }
    auto end = std::chrono::high_resolution_clock::now();
    latency = std::chrono::duration<double, std::milli>(end.time_since_epoch()).count() - capture_moment_ms_;
}
inline const char* getEncodingFromMatType(int conversion_type) {
    switch (conversion_type) {
        case cvt_YUV2GRAY: return sensor_msgs::image_encodings::MONO8;
        case cvt_YUV2BGR: return sensor_msgs::image_encodings::BGR8;
        case cvt_YUV2RGB: return sensor_msgs::image_encodings::RGB8;
    }
}
inline void copyMatToImageMsg(const cv::Mat& mat, sensor_msgs::msg::Image& msg, int converion_type = cvt_YUV2GRAY) {
    msg.height = mat.rows;
    msg.width = mat.cols;
    msg.encoding = getEncodingFromMatType(converion_type);
    msg.is_bigendian = false;
    msg.step = mat.cols * mat.elemSize();
    msg.data.assign(mat.data, mat.data + mat.total() * mat.elemSize());
}
void SyncStereoConsumerThread::get_image_msg(sensor_msgs::msg::Image& msg, double& latency) const {
    std::lock_guard<std::mutex> lock(frame_mutex_);

    if (assembled_mode) {
        cv::Mat resized_mat;
        cv::resize(this->concatFrame, resized_mat, cv::Size(1280, 480));
        copyMatToImageMsg(resized_mat, msg, conversion_type);
    }
    else {
        // If not in assembled mode, concatenate left and right frames
        cv::hconcat(this->leftFrame, this->rightFrame, concatFrame);
        cv::Mat resized_mat;
        cv::resize(this->concatFrame, resized_mat, cv::Size(1280, 480));

        copyMatToImageMsg(resized_mat, msg, conversion_type);
    }
    auto end = std::chrono::high_resolution_clock::now();
    latency = std::chrono::duration<double, std::milli>(end.time_since_epoch()).count() - capture_moment_ms_;
}

//====================================================================
//==================FRAME ACCQUIRE ACTION=============================
//====================================================================



CudaFrameAcquire::CudaFrameAcquire(CUeglStreamConnection& connection)
                                   : m_connection(connection)
                                   , m_stream(NULL), m_resource(0) {
  CUresult result = cuEGLStreamConsumerAcquireFrame(&m_connection, &m_resource, &m_stream, -1);
  if (result == CUDA_SUCCESS) {
    cuGraphicsResourceGetMappedEglFrame(&m_frame, m_resource, 0, 0);
  }
    else printf("Error when consumer acquire Frame (%s)", getCudaErrorString(result));

}

CudaFrameAcquire::~CudaFrameAcquire() {
  if (m_resource) {
    cuEGLStreamConsumerReleaseFrame(&m_connection, m_resource, &m_stream);
  }
}
bool CudaFrameAcquire::convert2CPU(bool leftFrame, int conversionType) {
    if (!hasValidFrame())
        ORIGINATE_ERROR("No valid frame to convert");
  CUDA_RESOURCE_DESC cudaResourceDesc;
  memset(&cudaResourceDesc, 0, sizeof(cudaResourceDesc));
  cudaResourceDesc.resType = CU_RESOURCE_TYPE_ARRAY;

  cudaResourceDesc.res.array.hArray = m_frame.frame.pArray[0];
  CUsurfObject cudaSurfObj1 = 0;
  CUresult cuResult = cuSurfObjectCreate(&cudaSurfObj1, &cudaResourceDesc);
  if (cuResult != CUDA_SUCCESS) {
    ORIGINATE_ERROR("Unable to create surface object 1 (%s)", getCudaErrorString(cuResult));
  }
  
  cudaResourceDesc.res.array.hArray = m_frame.frame.pArray[1];
  CUsurfObject cudaSurfObj2 = 0;
  cuResult = cuSurfObjectCreate(&cudaSurfObj2, &cudaResourceDesc);
  if (cuResult != CUDA_SUCCESS) {
    ORIGINATE_ERROR("Unable to create surface object 2 (%s)", getCudaErrorString(cuResult));
  }

  float delta = convert(cudaSurfObj1, cudaSurfObj2, m_frame.width, m_frame.height, oBuffer, conversionType);
  cuSurfObjectDestroy(cudaSurfObj1);
  cuSurfObjectDestroy(cudaSurfObj2);
    

  return true;
}

bool CudaFrameAcquire::generateHistogram(unsigned int histogramData[HISTOGRAM_BINS], uint64_t image_size,
                                                        float *curExposureLevel)
{
    if (!m_resource || !histogramData || !curExposureLevel)
        ORIGINATE_ERROR("Invalid state or output parameters");

    if (m_frame.cuFormat != CU_AD_FORMAT_UNSIGNED_INT8)
    {
        ORIGINATE_ERROR("Histogram supports only 8-bit unsigned int formats");
    }

    const bool isYuv = isCudaFormatYUV(m_frame.eglColorFormat);
    const bool isGray = (m_frame.numChannels == 1U);
    const bool isRgb = (m_frame.numChannels == 3U);
    const bool isRgba = (m_frame.numChannels == 4U);
    if (!isYuv && !isGray && !isRgb && !isRgba)
    {
        ORIGINATE_ERROR("Unsupported frame format for histogram (eglColorFormat=%d, channels=%u)",
                        static_cast<int>(m_frame.eglColorFormat),
                        m_frame.numChannels);
    }

    unsigned int histogramChannels = HISTOGRAM_GRAY;
    if (isRgb) histogramChannels = HISTOGRAM_RGB;
    else if (isRgba) histogramChannels = HISTOGRAM_RGBA;

    // Create surface from luminance channel.
    CUDA_RESOURCE_DESC cudaResourceDesc;
    memset(&cudaResourceDesc, 0, sizeof(cudaResourceDesc));
    cudaResourceDesc.resType = CU_RESOURCE_TYPE_ARRAY;
    cudaResourceDesc.res.array.hArray = m_frame.frame.pArray[0];
    CUsurfObject cudaSurfObj = 0;
    CUresult cuResult = cuSurfObjectCreate(&cudaSurfObj, &cudaResourceDesc);
    if (cuResult != CUDA_SUCCESS)
    {
        ORIGINATE_ERROR("Unable to create the surface object (CUresult %s)",
                         getCudaErrorString(cuResult));
    }

    // Generated the histogram.
    float execute_time = histogram(cudaSurfObj, m_frame.width, m_frame.height, histogramData, histogramChannels);
    (void)execute_time;
    uint64_t sum = 0;
    uint64_t currentBin;
    uint64_t count = 0;

    for (currentBin = 0; currentBin < HISTOGRAM_BINS; currentBin++)
    {
        sum += histogramData[currentBin] * (currentBin);
        count += histogramData[currentBin];
    }
    // // Visualize histogram:
    // const int VIS_W = 512;
    // const int VIS_H = 300;
    // const int MARGIN = 40;
    // const int TOTAL_H = VIS_H + MARGIN * 2;
    // const int TOTAL_W = VIS_W + MARGIN * 2;

    // cv::Mat canvas(TOTAL_H, TOTAL_W, CV_8UC3, cv::Scalar(30, 30, 30));

    // // Tìm max bin (bỏ qua bin 0 và bin 255 để scale đẹp hơn)
    // unsigned int max_count = 0;
    // for (unsigned int i = 1; i < HISTOGRAM_BINS - 1; i++)
    // {
    //     if (histogramData[i] > max_count) max_count = histogramData[i];
    // }
    // if (max_count == 0) max_count = 1;

    // float bar_w = (float)VIS_W / HISTOGRAM_BINS;

    // // Vẽ histogram bars
    // for (unsigned int i = 0; i < HISTOGRAM_BINS; i++)
    // {
    //     int bar_h = std::min((int)((uint64_t)histogramData[i] * VIS_H / max_count), VIS_H);
    //     int x = MARGIN + (int)(i * bar_w);
    //     int w = std::max((int)bar_w, 1);

    //     // Màu gradient: tối→sáng theo bin
    //     int c = i * 255 / (HISTOGRAM_BINS - 1);
    //     cv::Scalar color(c, c, c);

    //     cv::rectangle(canvas,
    //                     cv::Point(x, MARGIN + VIS_H - bar_h),
    //                     cv::Point(x + w, MARGIN + VIS_H),
    //                     color, cv::FILLED);
    // }

    // // Vẽ đường mean
    // float mean_val = (count > 0) ? (float)sum / count : 0.0f;
    // int mean_x = MARGIN + (int)(mean_val * VIS_W / (HISTOGRAM_BINS - 1));
    // cv::line(canvas,
    //             cv::Point(mean_x, MARGIN),
    //             cv::Point(mean_x, MARGIN + VIS_H),
    //             cv::Scalar(0, 0, 255), 2);  // Đỏ = mean

    // // Vẽ đường target
    // int target_x = MARGIN + (int)(TARGET_EXPOSURE_LEVEL * VIS_W);
    // cv::line(canvas,
    //             cv::Point(target_x, MARGIN),
    //             cv::Point(target_x, MARGIN + VIS_H),
    //             cv::Scalar(0, 255, 0), 2);  // Xanh lá = target

    // // Annotations
    // char text[128];
    // snprintf(text, sizeof(text), "Mean=%.1f (%.3f)", mean_val, mean_val / (HISTOGRAM_BINS - 1));
    // cv::putText(canvas, text, cv::Point(mean_x + 5, MARGIN + 15),
    //             cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

    // snprintf(text, sizeof(text), "Target=%.3f", TARGET_EXPOSURE_LEVEL);
    // cv::putText(canvas, text, cv::Point(target_x + 5, MARGIN + 30),
    //             cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);

    // snprintf(text, sizeof(text), "Pixels=%lu  ExpLv=%.4f",
    //             (unsigned long)count, (float)sum / (float)(HISTOGRAM_BINS * image_size));
    // cv::putText(canvas, text, cv::Point(MARGIN, TOTAL_H - 10),
    //             cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);

    // // Axis labels
    // cv::putText(canvas, "0", cv::Point(MARGIN, MARGIN + VIS_H + 15),
    //             cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(150, 150, 150), 1);
    // cv::putText(canvas, "255", cv::Point(MARGIN + VIS_W - 20, MARGIN + VIS_H + 15),
    //             cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(150, 150, 150), 1);

    // cv::imshow("Histogram Debug", canvas);
    *curExposureLevel = (float) sum / (float)(HISTOGRAM_BINS*image_size);
    // Destroy surface.
    cuSurfObjectDestroy(cudaSurfObj);

    return true;
}

bool CudaFrameAcquire::hasValidFrame() const
{
    return m_resource && checkFormat();
}

const uint8_t* CudaFrameAcquire::getOutputBuffer() const
{
    return m_outputBuffer.empty() ? nullptr : m_outputBuffer.data();
}

size_t CudaFrameAcquire::getFrameSize() const
{
    return m_frameSize;
}

bool CudaFrameAcquire::checkFormat() const
{
    if (!isCudaFormatYUV(m_frame.eglColorFormat))
    {
        ORIGINATE_ERROR("Only YUV color formats are supported");
    }
    if (m_frame.cuFormat != CU_AD_FORMAT_UNSIGNED_INT8)
    {
        ORIGINATE_ERROR("Only 8-bit unsigned int formats are supported");
    }
    return true;
}

}