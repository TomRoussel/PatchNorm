#include <iostream>
#include <math.h>
#include "patchnorm_cuda.h"

using std::cout; 
using std::endl;

// TODO: optimize by using integrated image
// Strategy would be: integrate image, and integrate it again with squared pixel values
// I and I2 respectively, N is amount of pixels
// mean= (I(br) - I(tl)) / N
// var = ((I2(br) - I2(tl)) - mean^2*N) / (N-1)

////// CUDA kernels //////

__global__
void CU_patchnorm(uchar *pixel, float *out, int h, int w, int neighb, size_t pitch_in, size_t pitch_out) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    int rad = (neighb - 1)/2;
    for(int idx=index; idx < w*h; idx+=stride) {
        int x = idx % w;
        int y = idx / w;

        int xinner_init = x - rad;
        xinner_init = ((xinner_init < 0) ? 0 : xinner_init);
        xinner_init = ((xinner_init >= w) ? w-1 : xinner_init);
        int yinner_init = y - rad;
        yinner_init = ((yinner_init < 0) ? 0 : yinner_init);
        yinner_init = ((yinner_init >= w) ? w-1 : yinner_init);

        // Get mean
        float mean = 0;
        for(int xinner = xinner_init; xinner < xinner_init + neighb; xinner++) {
            for(int yinner = yinner_init; yinner < yinner_init + neighb; yinner++) {
                uchar* pixelval = (uchar*)((char*)pixel + yinner * pitch_in) + xinner;
                mean += (float)*pixelval;
            }
        }
        mean /= neighb * neighb;
        // Calculate variance
        float var = 0;
        for(int xinner = xinner_init; xinner < xinner_init + neighb; xinner++) {
            for(int yinner = yinner_init; yinner < yinner_init + neighb; yinner++) {
                uchar* pixelval = (uchar*)((char*)pixel + yinner * pitch_in) + xinner;
                var += (float) (*pixelval - mean) * (*pixelval - mean);
            }
        }
        var /= neighb * neighb;
        // Add small value to denominator (to prevent huge values around zero)
        float stdev = sqrt(var) + 1e-3;
        
        // Patchnorm the current pixel
        uchar* pixelval = (uchar*)((char*)pixel + y* pitch_in) + x;
        float* outpixel = (float *)((char*)out + y * pitch_out) + x;
        *outpixel = (*pixelval - mean)/stdev;

        /* printf("x: %d, y: %d, xinner: %d, yinner: %d, mean: %f, stdev: %f, pixelval: %d, value: %f\n", x, y, xinner_init, yinner_init, mean, stdev, *pixelval, *outpixel); */
    }
}

////// C++ code //////

PatchNormCuda::PatchNormCuda(int width, int height): width(width), height(height) {
    // Initialize CUDA memory
    cudaMallocPitch(&in_ptr, &pitch_in, width, height);
    cudaMallocPitch(&out_ptr, &pitch_out, width * sizeof(float), height);    

    blockSize = 512;
    numBlocks = (width * height + blockSize - 1) / blockSize;
}

PatchNormCuda::~PatchNormCuda() {
    cudaFree(in_ptr);
    cudaFree(out_ptr);
}

void PatchNormCuda::compute(cv::Mat& in, cv::Mat& out, int neighb) {
    if(!in.isContinuous()) {
        cout << "NOT CONTINUOUS" << endl;
        cout << "IS BAD" << endl;
        exit(1);
    }

    cudaMemcpy2D(in_ptr, pitch_in, in.ptr(), width, width, height, cudaMemcpyHostToDevice);
    
    CU_patchnorm<<<numBlocks,blockSize>>>(in_ptr, out_ptr, height, width, neighb, pitch_in, pitch_out);

    float* out_ptr_host = new float[width * height]();
    cudaMemcpy2D(out_ptr_host, width*sizeof(float), out_ptr, pitch_out, width * sizeof(float), height, cudaMemcpyDeviceToHost);

    out = cv::Mat(height, width, CV_32FC1, out_ptr_host);
    // don't free out_ptr_host, data is managed by the new Mat
    /* free(out_ptr_host); */
}
