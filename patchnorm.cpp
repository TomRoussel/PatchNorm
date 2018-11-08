#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "patchnorm_cpu.hpp"
#include "patchnorm_cuda.h"
#include <chrono>

using namespace std;
using namespace cv;

string fn = "test2.png";

int main(int argc, char** argv) {
    if(argc != 2) {
        cout << "Usage: patchnorm PATH_TO_IMAGE" << endl;
        exit(1);
    }
	Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat pn, pn2;
    
    cout << "Running patch norm on CPU" << endl;
    auto t1 = std::chrono::steady_clock::now();
	patchnorm(img, pn, 17);
	normalize(pn, pn, 255, 0, NORM_MINMAX);
    pn.convertTo(pn, CV_8UC1);
    
    auto t2 = std::chrono::steady_clock::now();
    cout << "Running patch norm on GPU" << endl;
    patchnorm_cuda(img, pn2, 17);
    normalize(pn2, pn2, 255, 0, NORM_MINMAX);
    pn2.convertTo(pn2, CV_8UC1);
    /* cout << pn2 << endl; */
    auto t3 = std::chrono::steady_clock::now();

    auto dur_cpu = chrono::duration_cast<chrono::microseconds>(t2-t1).count() / 1000.f;
    auto dur_gpu = chrono::duration_cast<chrono::microseconds>(t3-t2).count() / 1000.f;

    printf("Calculating patch norm on cpu took %f milliseconds.\nCalculating patch norm on gpu took %f milliseconds\n", dur_cpu, dur_gpu);
	/* imshow("Patchnormed", pn); */
    imshow("Patchnormed cuda", pn2);
	while (waitKey(0) != 'q');
}
