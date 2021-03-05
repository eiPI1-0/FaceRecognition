#include <iostream>
#include "FaceRecognition.h"

using namespace cv;

int main(int argc, char *argv[])
{
    float recogn_thresh=0.8;
    const int rfactor = 4;
    Size img_shape={160, 120}, display_shape=img_shape*rfactor, features_model_shpae={96, 96};
    String detection_model = "./models/YuFaceDetectNet.onnx";
    String recognize_model = "./models/openface.nn4.small2.v1.t7";
    String data_path = "./data/";

    Detector detector(detection_model, img_shape);
    FeatureExtractor extractor(recognize_model, features_model_shpae);
    FeatureDatabase<NormSimilarity> database(data_path, detector,extractor,
                                             display_shape, rfactor, recogn_thresh);

    cv::VideoCapture cap;
    cv::Mat im;
    cv::TickMeter cvtm;
    std::vector<cv::String> output_names = {"loc", "conf", "iou"};
    std::vector<cv::Mat> faces, features;

    String title = "FaceRecognition@eipi10";

    if (isdigit(argv[1][0])) {
        cap.open(argv[1][0] - '0');
        if (!cap.isOpened()) {
            std::cerr << "Cannot open the camera." << std::endl;
            return 0;
        }
    }

    if (cap.isOpened()) {
        while (true) {
            cap >> im;
            cvtm.start();

            if (features_from_img(im, detector, extractor, features,
                                  display_shape, rfactor)) {
//                std::cout << detector.size() << " faces found.\n";
                int i=0;
                float max_similar;
                std::string max_name;
                for (const auto& feature : features) {
                    if (database.select_max_mean_similarity(feature, max_similar,
                                                            max_name)) {
                    } else {
                        max_name = std::string("Unknown");
                    }
                    cv::putText(im, max_name, detector.faces()[i].bbox.top_left*rfactor,
                                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                cv::Scalar(0, 255, 0), 1);
                    ++i;
                }
            } else {
                std::cout << "No faces found." << std::endl;
            }

            cvtm.stop();

            std::string timeLabel = cv::format("Inference time: %.2f ms", cvtm.getTimeMilli());
            cv::putText(im, timeLabel, cv::Point(0, 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

            draw(im, detector.faces() , rfactor);

            cvtm.reset();
            cv::imshow(title, im);
            if ((cv::waitKey(1) & 0xFF) == 27)
                break;
        }
    }

    return 0;
}
