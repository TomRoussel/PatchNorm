#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "patchnorm_cuda.h"
#include <experimental/filesystem>

#define NEIGHB 17

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;

int main(int argc, char**argv) {
    if (argc < 3) {
        cout << "Usage patchnorm_convert output_folder [images...]" << endl;
        exit(1);
    }

    fs::path outputfolder = argv[1];
    if(!fs::is_directory(outputfolder)) {
        cout << outputfolder << " Is not a directory!" << endl;
        exit(1);
    }

    PatchNormCuda * normalizer = NULL;
    Size last_size(0,0);
    for(int idx=2; idx < argc; idx++) {
        fs::path imgfn = argv[idx];
        fs::path outfn = outputfolder / imgfn.filename();

        Mat img = imread(imgfn.string(), CV_LOAD_IMAGE_GRAYSCALE);
        Mat pn;
        if((!normalizer) | (img.size() != last_size)) {
            // Lazily initialize the normalizer
            // If the image size changed, the normalizer also needs to be reinitialized
            delete normalizer;
            normalizer = new PatchNormCuda(img.size().width, img.size().height);
        }

        last_size = img.size();

        normalizer->compute(img, pn, 17);

        normalize(pn, pn, 255, 0, NORM_MINMAX);
        pn.convertTo(pn, CV_8UC1);

        imwrite(outfn.string(), pn);
        printf("Finished %d/%d\r", idx-1, argc-2);
        cout << flush;
    }
    cout << endl;
}
