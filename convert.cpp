#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "patchnorm_cuda.h"

#define NEIGHB 17

using namespace std;
using namespace cv;

int main(int argc, char**argv) {
    if (argc != 2) {
        cout << "Usage patchnorm_convert path_to_image" << endl;
        exit(1);
    }

    Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat pn;

    PatchNormCuda normalizer(img.size().width, img.size().height);

    normalizer.compute(img, pn, 17);

    normalize(pn, pn, 255, 0, NORM_MINMAX);
    pn.convertTo(pn, CV_8UC1);

    imwrite("pn.png", pn);
}
