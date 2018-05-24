#include "YOLO.h"

int main(int argc, char* argv[]) {

    YOLO yolo(argv[1]);
    cv::Mat input = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);

    yolo.pred(input);

    return 0;
}