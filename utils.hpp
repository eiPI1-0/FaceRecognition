#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <vector>
#include <algorithm>
#include "face_def.hpp"
#include "opencv2/opencv.hpp"

void nms(std::vector<Face>& dets,
         const float thresh = 0.3);

// 保持长宽比缩放到 size 内并居中补边（默认黑边）。输出尺寸恒等于 size，
// 所以 draw()/putText() 用 rfactor 映射检测坐标（160x120 空间）依然成立：
// 补边后的图正好是 input_shape * rfactor。返回新 Mat，不与输入共享数据。
cv::Mat letterbox(const cv::Mat& img, const cv::Size& size,
                  const cv::Scalar& border = cv::Scalar(0, 0, 0));
void draw(cv::Mat& img,
          const std::vector<Face>& faces,
	  const int rfactor = 5);

#endif
