// YOLO.h
// @author : Jinwoo Jang
// @email : real.longrain@gmail.com
// Inference module using YOLO model written python3-tensorflow
// Used graph and weights : https://github.com/nilboy/tensorflow-yolo

#ifndef TF_TEST_YOLO_H
#define TF_TEST_YOLO_H

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <tuple>
#include <algorithm>

#include "Eigen/Dense"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include "opencv2/opencv.hpp"

const int IMG_SIZE = 448;
const int CELL_SIZE = 7;
const int BOXES_PER_CELL = 2;
const int NUM_CLASSES = 20;
const int BOUNDARY1 = CELL_SIZE * CELL_SIZE * NUM_CLASSES;
const int BOUNDARY2 = BOUNDARY1 + CELL_SIZE * CELL_SIZE * BOXES_PER_CELL;
const float THRESHOLD = 0.2;
const float IOU_THRESHOLD = 0.5;

const std::string FEED_NAME = "images";
const std::string OUTPUT_NODE_NAME = "yolo/fc_36/BiasAdd";


class YOLO {
private:
    tensorflow::Session* session;
    tensorflow::Status status;
    tensorflow::GraphDef graphDef;

public:
    YOLO(const std::string model_path) {
        status = NewSession(tensorflow::SessionOptions(), &session);
        if (!status.ok()) { std::cout << status.ToString() << std::endl; };
        status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), model_path, &graphDef);
        if (!status.ok()) { std::cout << status.ToString() << std::endl; };
        status = session->Create(graphDef);
        if (!status.ok()) { std::cout << status.ToString() << std::endl; };
    }

private:
    void img_to_tensor(cv::Mat& input, tensorflow::Tensor& input_tensor) {
        cv::resize(input, input, cv::Size(IMG_SIZE, IMG_SIZE), 0, 0, CV_INTER_LINEAR);
        cv::cvtColor(input, input, cv::COLOR_BGR2RGB);

        auto input_tensor_mapped = input_tensor.tensor<float, 4>();

        for(int y = 0; y < input.size().height; y++) {
            for(int x = 0; x < input.size().width; x++) {
                for(int c = 0; c < 3; c++) {
                    input_tensor_mapped(0, y, x, c) = (unsigned char)input.at<cv::Vec3b>(y, x)[c] / 255.0 * 2.0 - 1.0;
                }
            }
        }
    }

    std::vector<tensorflow::Tensor> run_graph(const std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs) {
        std::vector<tensorflow::Tensor> outputs;
        status = session->Run(inputs, {OUTPUT_NODE_NAME}, {}, &outputs);
        if (!status.ok()) { std::cout << status.ToString() << std::endl; exit(1); }
        else { std::cout << "load success" << std::endl; }
        return outputs;
    }

    double iou(std::tuple<float, float, float, float>& box1, std::tuple<float, float, float, float>& box2) {
        double tb = std::min(std::get<0>(box1) + 0.5 * std::get<2>(box1), std::get<0>(box2) + 0.5 * std::get<2>(box2)) -
                   std::max(std::get<0>(box1) - 0.5 * std::get<2>(box1), std::get<0>(box2) - 0.5 * std::get<2>(box2));
        double lr = std::min(std::get<1>(box1) + 0.5 * std::get<3>(box1), std::get<1>(box2) + 0.5 * std::get<3>(box2)) -
                   std::max(std::get<1>(box1) - 0.5 * std::get<3>(box1), std::get<1>(box2) - 0.5 * std::get<3>(box2));
        double inter;
        if (tb < 0 || lr < 0) inter = 0.0;
        else inter = tb * lr;
        return inter / (std::get<2>(box1) * std::get<3>(box1) + std::get<2>(box2) * std::get<3>(box2) - inter);
    }

    std::vector<std::tuple<int, float, float, float, float, float>> interpret_output(const std::vector<tensorflow::Tensor>& output) {
        auto result = output[0].tensor<float, 2>();
        std::vector<float> out_vector; // 1470

        for (int i = 0; i < result.size(); i++) {
            out_vector.push_back(result(0, i));
        }

        // PROBS
        float probs[CELL_SIZE][CELL_SIZE][BOXES_PER_CELL][NUM_CLASSES];
        memset(probs, 0, sizeof(float) * CELL_SIZE * CELL_SIZE * BOXES_PER_CELL * NUM_CLASSES);

        // CLASS_PROBS
        float class_probs[CELL_SIZE][CELL_SIZE][NUM_CLASSES];
        for (int c1 = 0; c1 < CELL_SIZE; c1++) {
            for (int c2 = 0; c2 < CELL_SIZE; c2++) {
                for (int nc = 0; nc < NUM_CLASSES; nc++) {
                    class_probs[c1][c2][nc] = out_vector[(c1 * CELL_SIZE * NUM_CLASSES) + (c2 * NUM_CLASSES) + nc];
                }

            }
        }

        // SCALES
        float scales[CELL_SIZE][CELL_SIZE][BOXES_PER_CELL];
        for (int c1 = 0; c1 < CELL_SIZE; c1++) {
            for (int c2 = 0; c2 < CELL_SIZE; c2++) {
                for (int b = 0; b < BOXES_PER_CELL; b++) {
                    scales[c1][c2][b] = out_vector[BOUNDARY1 + (c1 * CELL_SIZE * BOXES_PER_CELL) + (c2 * BOXES_PER_CELL) + b];
                }
            }
        }

        // BOXES
        float boxes[CELL_SIZE][CELL_SIZE][BOXES_PER_CELL][4];
        for (int c1 = 0; c1 < CELL_SIZE; c1++) {
            for (int c2 = 0; c2 < CELL_SIZE; c2++) {
                for (int b = 0; b < BOXES_PER_CELL; b++) {
                    for (int p = 0; p < 4; p++) {
                        int idx = BOUNDARY2 + (c1 * CELL_SIZE * BOXES_PER_CELL * 4) + (c2 * BOXES_PER_CELL * 4) + (b * 4) + p;
                        boxes[c1][c2][b][p] = out_vector[idx];
                    }
                }
            }
        }

        // OFFSET
        float offset[CELL_SIZE][CELL_SIZE][BOXES_PER_CELL];
        for (int c1 = 0; c1 < CELL_SIZE; c1++) {
            for (int c2 = 0; c2 < CELL_SIZE; c2++) {
                offset[c1][c2][0] = c2;
                offset[c1][c2][1] = c2;
            }
        }

        // boxes[:, :, :, 0] += offset
        for (int c1 = 0; c1 < CELL_SIZE; c1++) {
            for (int c2 = 0; c2 < CELL_SIZE; c2++) {
                for (int b = 0; b < BOXES_PER_CELL; b++) {
                    boxes[c1][c2][b][0] += offset[c1][c2][b];
                }
            }
        }



        // boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        for (int c1 = 0; c1 < CELL_SIZE; c1++) {
            for (int c2 = 0; c2 < CELL_SIZE; c2++) {
                for (int b = 0; b < BOXES_PER_CELL; b++) {
                    boxes[c1][c2][b][1] += offset[c2][c1][b];
                }
            }
        }

        // boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        for (int c1 = 0; c1 < CELL_SIZE; c1++) {
            for (int c2 = 0; c2 < CELL_SIZE; c2++) {
                for (int b = 0; b < BOXES_PER_CELL; b++) {
                    boxes[c1][c2][b][0] = 1.0 * boxes[c1][c2][b][0] / 7.f;
                    boxes[c1][c2][b][1] = 1.0 * boxes[c1][c2][b][1] / 7.f;
                }
            }
        }

        // boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])
        for (int c1 = 0; c1 < CELL_SIZE; c1++) {
            for (int c2 = 0; c2 < CELL_SIZE; c2++) {
                for (int b = 0; b < BOXES_PER_CELL; b++) {
                    boxes[c1][c2][b][2] = boxes[c1][c2][b][2] * boxes[c1][c2][b][2];
                    boxes[c1][c2][b][3] = boxes[c1][c2][b][3] * boxes[c1][c2][b][3];
                }
            }
        }

        // boxes *= self.image_size
        for (int c1 = 0; c1 < CELL_SIZE; c1++) {
            for (int c2 = 0; c2 < CELL_SIZE; c2++) {
                for (int b = 0; b < BOXES_PER_CELL; b++) {
                    for (int p = 0; p < 4; p++) {
                        boxes[c1][c2][b][p] *= IMG_SIZE;
                    }
                }
            }
        }

        for (int c1 = 0; c1 < CELL_SIZE; c1++) {
            for (int c2 = 0; c2 < CELL_SIZE; c2++) {
                for (int b = 0; b < BOXES_PER_CELL; b++) {
                    for (int nc = 0; nc < NUM_CLASSES; nc++) {
                        probs[c1][c2][b][nc] = class_probs[c1][c2][nc] * scales[c1][c2][b];
                    }
                }
            }
        }

        // filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        // filter_mat_boxes = np.nonzero(filter_mat_probs)
        // boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        bool filter_mat_probs[CELL_SIZE][CELL_SIZE][BOXES_PER_CELL][NUM_CLASSES];
        std::vector<float> probs_filtered;
        std::vector<std::tuple<int, int, int, int>> filter_mat_boxes;
        for (int c1 = 0; c1 < CELL_SIZE; c1++) {
            for (int c2 = 0; c2 < CELL_SIZE; c2++) {
                for (int b = 0; b < BOXES_PER_CELL; b++) {
                    for (int nc = 0; nc < NUM_CLASSES; nc++) {
                        if (probs[c1][c2][b][nc] >= THRESHOLD) {
                            filter_mat_boxes.push_back(std::tuple<int, int, int, int>(c1, c2, b, nc));
                            probs_filtered.push_back(probs[c1][c2][b][nc]); // ok
                            filter_mat_probs[c1][c2][b][nc] = true;
                        } else {
                            filter_mat_probs[c1][c2][b][nc] = false;
                        }
                    }
                }
            }
        }


        std::vector<std::tuple<float, float, float, float>> boxes_filtered;
        for (auto p : filter_mat_boxes) {

            int c1 = std::get<0>(p);
            int c2 = std::get<1>(p);
            int b = std::get<2>(p);

            boxes_filtered.push_back(
                    std::tuple<float, float, float, float>(boxes[c1][c2][b][0],
                                                           boxes[c1][c2][b][1],
                                                           boxes[c1][c2][b][2],
                                                           boxes[c1][c2][b][3])
            );
        }

        int argmax[CELL_SIZE][CELL_SIZE][BOXES_PER_CELL];
        memset(argmax, 0, sizeof(int) * CELL_SIZE * CELL_SIZE * BOXES_PER_CELL);
        for (int c1 = 0; c1 < CELL_SIZE; c1++) {
            for (int c2 = 0; c2 < CELL_SIZE; c2++) {
                for (int b = 0; b < BOXES_PER_CELL; b++) {
                    for (int nc = 0; nc < NUM_CLASSES; nc++) {
                        if (filter_mat_probs[c1][c2][b][nc]) {
                            argmax[c1][c2][b] = nc;
                            break;
                        }
                    }
                }
            }
        }

        std::vector<int> class_num_filtered; // ok
        for (auto p : filter_mat_boxes) {
            int c1 = std::get<0>(p);
            int c2 = std::get<1>(p);
            int b = std::get<2>(p);
            class_num_filtered.push_back(argmax[c1][c2][b]);
        }

        std::vector<int> argsort;
        std::vector<float> sorted_probs(probs_filtered.begin(), probs_filtered.end());
        std::sort(sorted_probs.begin(), sorted_probs.end());

        for (auto sv : sorted_probs) {
            for (int i = 0; i < probs_filtered.size(); i++) {
                if (probs_filtered[i] == sv) {
                    argsort.push_back(i);
                }
            }
        }

        argsort = std::vector<int>(argsort.rbegin(), argsort.rend());


        std::vector<std::tuple<float, float, float, float>> argsort_boxes_filtered;
        std::vector<float> argsort_probs_filtered; // ok
        std::vector<int> argsort_class_num_filtered;
        for (auto idx : argsort) {
            argsort_boxes_filtered.push_back(boxes_filtered[idx]);
            argsort_probs_filtered.push_back(probs_filtered[idx]);
            argsort_class_num_filtered.push_back(class_num_filtered[idx]);
        }

        for (int i = 0; i < argsort_boxes_filtered.size(); i++) {
            if (argsort_probs_filtered[i] == 0) continue;
            for (int j = i + 1; j < argsort_boxes_filtered.size(); j++) {

                if (iou(argsort_boxes_filtered[i], argsort_boxes_filtered[j]) > IOU_THRESHOLD) {
                    argsort_probs_filtered[j] = 0.0;
                }
            }
        }


        std::vector<std::tuple<float, float, float, float>> iousort_boxes_filtered;
        std::vector<float> iousort_probs_filtered;
        std::vector<int> iousort_class_num_filtered;

        for(int i = 0; i < argsort_probs_filtered.size(); i++) {
            if (argsort_probs_filtered[i] > 0.0) {
                iousort_boxes_filtered.push_back(argsort_boxes_filtered[i]);
                iousort_probs_filtered.push_back(argsort_probs_filtered[i]);
                iousort_class_num_filtered.push_back(argsort_class_num_filtered[i]);
            }
        }

        std::vector<std::tuple<int, float, float, float, float, float>> resolved;

        for (int i = 0; i < iousort_probs_filtered.size(); i++) {
            resolved.push_back(std::tuple<int, float, float, float, float, float>(
                    iousort_class_num_filtered[i],
                    std::get<0>(iousort_boxes_filtered[i]),
                    std::get<1>(iousort_boxes_filtered[i]),
                    std::get<2>(iousort_boxes_filtered[i]),
                    std::get<3>(iousort_boxes_filtered[i]),
                    iousort_probs_filtered[i]));
            std::cout << iousort_class_num_filtered[i] << " "
                    << std::get<0>(iousort_boxes_filtered[i]) << " "
                    << std::get<1>(iousort_boxes_filtered[i]) << " "
                    << std::get<2>(iousort_boxes_filtered[i]) << " "
                    << std::get<3>(iousort_boxes_filtered[i]) << " "
                    << iousort_probs_filtered[i] << std::endl;
        }

        return resolved;
    }

public:
    void pred(cv::Mat input) {
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, IMG_SIZE, IMG_SIZE, 3}));
        cv::Mat outimg = input.clone();
        img_to_tensor(input, input_tensor);
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
                { "images", input_tensor }
        };


        auto result = interpret_output(run_graph(inputs));
        for (auto r : result) {
            if (std::get<0>(r) != 14) continue;
            std::get<1>(r) *= (1.0 * outimg.size().width / IMG_SIZE);
            std::get<2>(r) *= (1.0 * outimg.size().height / IMG_SIZE);
            std::get<3>(r) *= (1.0 * outimg.size().width / IMG_SIZE);
            std::get<4>(r) *= (1.0 * outimg.size().height / IMG_SIZE);

            int x = int(std::get<1>(r));
            int y = int(std::get<2>(r));
            int w = int(std::get<3>(r) / 2);
            int h = int(std::get<4>(r) / 2);

            cv::rectangle(outimg, cv::Point(x - w, y - h), cv::Point(x + w, y + h), cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("test", outimg);
        cv::waitKey();
    }
};

#endif //TF_TEST_YOLO_H
