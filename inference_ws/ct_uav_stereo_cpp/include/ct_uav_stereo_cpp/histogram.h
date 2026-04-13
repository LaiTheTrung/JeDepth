#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <cuda.h>

#define HISTOGRAM_BINS 64

enum HistogramPixelChannels
{
    HISTOGRAM_GRAY = 1,
    HISTOGRAM_RGB = 3,
    HISTOGRAM_RGBA = 4,
};

extern float histogram(CUsurfObject surface, unsigned int width, unsigned int height,
    unsigned int *histogram, unsigned int channels = HISTOGRAM_GRAY);

#endif // HISTOGRAM_H
