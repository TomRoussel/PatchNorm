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

__global__
void CU_patchnorm(uchar *pixel, float *out, int h, int w, int neighb, size_t pitch_in, size_t pitch_out) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    int rad = (neighb - 1)/2;
    for(int x=index; x < w; x+=stride) {
        for(int y=0; y < h; y++) {
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
            float stdev = sqrt(var) + 1e-3;
            
            // Patchnorm the current pixel
            uchar* pixelval = (uchar*)((char*)pixel + y* pitch_in) + x;
            float* outpixel = (float *)((char*)out + y * pitch_out) + x;
            *outpixel = (*pixelval - mean)/stdev;


            /* printf("x: %d, y: %d, xinner: %d, yinner: %d, mean: %f, stdev: %f, pixelval: %d, value: %f\n", x, y, xinner_init, yinner_init, mean, stdev, *pixelval, *outpixel); */

        }
    }
}

void patchnorm_cuda(cv::Mat& in, cv::Mat& out, int neighb) {
    uchar* in_ptr;
    float* out_ptr;
    size_t pitch_in, pitch_out;
    cudaMallocPitch(&in_ptr, &pitch_in, in.size().width, in.size().height);
    cudaMallocPitch(&out_ptr, &pitch_out, in.size().width * sizeof(float), in.size().height);    

    if(!in.isContinuous()) {
        cout << "NOT CONTINUOUS" << endl;
        cout << "IS BAD" << endl;
        exit(1);
    }

    cudaMemcpy2D(in_ptr, pitch_in, in.ptr(), in.size().width, in.size().width, in.size().height, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (in.size().width + blockSize - 1) / blockSize;
    CU_patchnorm<<<numBlocks,blockSize>>>(in_ptr, out_ptr, in.size().height, in.size().width, neighb, pitch_in, pitch_out);

    float* out_ptr_host = new float[in.size().width * in.size().height]();
    cudaMemcpy2D(out_ptr_host, in.size().width*sizeof(float), out_ptr, pitch_out, in.size().width * sizeof(float), in.size().height, cudaMemcpyDeviceToHost);

    out = cv::Mat(in.size().height, in.size().width, CV_32FC1, out_ptr_host);

    cudaFree(in_ptr);
    cudaFree(out_ptr);

    free(out_ptr_host);

    /* for(int x=0; x<out.cols; x++) { */
    /*     for(int y=0; y<out.rows; y++) { */
    /*         printf("x: %d, y: %d, value: ", x, y); */
    /*         cout << out.at<float>(y,x) << endl; */
    /*     } */
    /* } */

}
