#include "utils.hpp"

void nms(std::vector<Face>& dets,
         const float thresh) {
    std::sort(dets.begin(), dets.end(), [](const Face& a, const Face& b) { return a.score > b.score; });

    std::vector<Face> kept;
    kept.reserve(dets.size());
    std::vector<bool> suppressed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        kept.push_back(dets[i]);

        const float area_i = dets[i].bbox.area();
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }

            float ix1 = std::max(dets[i].bbox.top_left.x, dets[j].bbox.top_left.x);
            float iy1 = std::max(dets[i].bbox.top_left.y, dets[j].bbox.top_left.y);
            float ix2 = std::min(dets[i].bbox.bottom_right.x, dets[j].bbox.bottom_right.x);
            float iy2 = std::min(dets[i].bbox.bottom_right.y, dets[j].bbox.bottom_right.y);

            float iw = ix2 - ix1 + 1;
            float ih = iy2 - iy1 + 1;
            if (iw <= 0 || ih <= 0) {
                continue;
            }
            float inter = iw * ih;

            const float area_j = dets[j].bbox.area();
            float iou = inter / (area_i + area_j - inter);
            if (iou > thresh) {
                suppressed[j] = true;
            }
        }
    }

    dets = std::move(kept);
}

void draw(cv::Mat& img,
          const std::vector<Face>& faces,
	  const int rfactor) {

    const int thickness = 2;
    const cv::Scalar bbox_color = {  0, 255,   0};
    const cv::Scalar text_color = {255, 255, 255};
    const std::vector<cv::Scalar> landmarks_color = {
        {255,   0,   0}, // left eye
        {  0,   0, 255}, // right eye
        {  0, 255, 255}, // mouth left
        {255, 255,   0}, // nose
        {  0, 255,   0}  // mouth right
    };

    auto point2f2point = [](cv::Point2f p, int rfactor = 1, bool shift = false) {
        return shift ? cv::Point(int(p.x*rfactor), int(p.y*rfactor)+12) : cv::Point(int(p.x*rfactor), int(p.y*rfactor));
    };
    for (auto i = 0; i < faces.size(); ++i) {
        // draw bbox
        cv::rectangle(img,
                      point2f2point(faces[i].bbox.top_left, rfactor),
                      point2f2point(faces[i].bbox.bottom_right, rfactor),
                      bbox_color,
                      thickness);
        // put score by the corner of bbox
        std::string str_score = std::to_string(faces[i].score);
        if (str_score.size() > 6) {
            str_score.erase(6);
        }
        cv::putText(img,
                    str_score,
                    point2f2point(faces[i].bbox.top_left, rfactor, true),
                    cv::FONT_HERSHEY_DUPLEX,
                    0.5, // Font scale
                    text_color);
        // draw landmarks
        const int radius = 2;
        cv::circle(img, point2f2point(faces[i].landmarks.left_eye, rfactor),    radius, landmarks_color[0], thickness);
        cv::circle(img, point2f2point(faces[i].landmarks.right_eye, rfactor),   radius, landmarks_color[1], thickness);
        cv::circle(img, point2f2point(faces[i].landmarks.mouth_left, rfactor),  radius, landmarks_color[2], thickness);
        cv::circle(img, point2f2point(faces[i].landmarks.nose_tip, rfactor),    radius, landmarks_color[3], thickness);
        cv::circle(img, point2f2point(faces[i].landmarks.mouth_right, rfactor), radius, landmarks_color[4], thickness);
    }
}
