#include <opencv2/core.hpp>

class PatchNormCuda {
    public:
        PatchNormCuda(int width, int height);
        ~PatchNormCuda();

        void compute(cv::Mat& in, cv::Mat& out, int neighb);
    private:
        uchar* in_ptr;
        float* out_ptr;
        size_t pitch_in, pitch_out;

        int width, height;
        int blockSize, numBlocks;
};

