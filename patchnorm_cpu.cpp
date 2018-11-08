#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "patchnorm_cpu.hpp"

using namespace cv;

void patchnorm(Mat in, Mat& out, int neighb) {
    in.convertTo(in, CV_32FC1);
    out = in.clone();
    Size dims = in.size();
    Rect full_region(0,0,dims.width, dims.height);
    int rad = (neighb - 1)/2;
    for(int x=0; x<dims.width; x++) {
        for(int y=0; y<dims.height; y++) {
            Rect roi(x-rad, y-rad, neighb, neighb);
            roi = roi & full_region;
            Mat mat_roi = in(roi);
            Scalar mean, stddev;

            meanStdDev(mat_roi, mean, stddev);
            /* std::cout << mean << " " << stddev << std::endl; */
            Mat tmp = (mat_roi - mean[0])/(stddev[0] + 1e-3);
            tmp.copyTo(out(roi));
        }
    }
}
