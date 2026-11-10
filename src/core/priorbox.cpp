#include "core/priorbox.hpp"

#include <string>

// 整个内核在一个命名空间里：`facerec::` 以上是库的边界，
// 也是 `api` 往上早就在用的那个名字（见 RFPLAN §6.1）。
namespace facerec {

PriorBox::PriorBox(const cv::Size& input_shape,
                   const cv::Size& output_shape) {
    // initialize
    in_w = input_shape.width;
    in_h = input_shape.height;
    out_w = output_shape.width;
    out_h = output_shape.height;

    cv::Size feature_map_2nd = {
        int(int((in_w+1)/2)/2), int(int((in_h+1)/2)/2)
    };
    cv::Size feature_map_3rd = {
        int(feature_map_2nd.width/2), int(feature_map_2nd.height/2)
    };
    cv::Size feature_map_4th = {
        int(feature_map_3rd.width/2), int(feature_map_3rd.height/2)
    };
    cv::Size feature_map_5th = {
        int(feature_map_4th.width/2), int(feature_map_4th.height/2)
    };
    cv::Size feature_map_6th = {
        int(feature_map_5th.width/2), int(feature_map_5th.height/2)
    };

    // feature_map_sizes.push_back(feature_map_2nd);
    feature_map_sizes.push_back(feature_map_3rd);
    feature_map_sizes.push_back(feature_map_4th);
    feature_map_sizes.push_back(feature_map_5th);
    feature_map_sizes.push_back(feature_map_6th);

    priors = generate_priors();
}

PriorBox::~PriorBox() {}

std::vector<BndBox_xywh> PriorBox::generate_priors() {
    std::vector<BndBox_xywh> anchors;
    for (auto i = 0; i < feature_map_sizes.size(); ++i) {
        cv::Size feature_map_size = feature_map_sizes[i];
        std::vector<float> min_size = min_sizes[i];

        for (auto _h = 0; _h < feature_map_size.height; ++_h) {
            for (auto _w = 0; _w < feature_map_size.width; ++_w) {
                for (auto j = 0; j < min_size.size(); ++j) {
                    float s_kx = min_size[j] / in_w;
                    float s_ky = min_size[j] / in_h;

                    float cx = (_w + 0.5) * steps[i] / in_w;
                    float cy = (_h + 0.5) * steps[i] / in_h;

                    // if (i == 3) {
                    //     std::cout << _h << " " << _w << std::endl;
                    // }

                    BndBox_xywh anchor = { {cx, cy}, s_kx, s_ky };
                    anchors.push_back(anchor);
                }
            }
        }
    }
    return anchors;
}

std::vector<Face> PriorBox::decode(const cv::Mat& loc,
                                   const cv::Mat& conf,
                                   const cv::Mat& iou,
                                   const float ignore_score) const {
    std::vector<Face> dets; // num * [x1, y1, x2, y2, x_re, y_re, x_le, y_le, x_ml, y_ml, x_n, y_n, x_mr, y_ml]

    // 头/先验网格的契约必须显式检查：返回空会被读成"没检测到脸"（与真漏检不可区分），
    // 而且契约破坏时下面的循环会越界读 loc_v/conf_v/iou_v（UB）。
    if (loc.type() != CV_32F || conf.type() != CV_32F || iou.type() != CV_32F ||
        !loc.isContinuous() || !conf.isContinuous() || !iou.isContinuous()) {
        CV_Error(cv::Error::StsBadArg,
                 "PriorBox::decode: loc/conf/iou must be continuous CV_32F vectors "
                 "(check detector.out_loc/out_conf/out_iou)");
    }
    if (loc.total() != priors.size() * 14 || conf.total() != priors.size() * 2 ||
        iou.total() != priors.size()) {
        CV_Error(cv::Error::StsBadArg,
                 "PriorBox::decode: prior grid has " + std::to_string(priors.size()) +
                 " anchors for detector input " + std::to_string(in_w) + "x" + std::to_string(in_h) +
                 ", the model returned loc=" + std::to_string(loc.total() / 14) +
                 " conf=" + std::to_string(conf.total() / 2) +
                 " iou=" + std::to_string(iou.total()) +
                 ": either detector.input_w/h disagrees with this export, or the model uses a"
                 " different head (per-level cls_*/obj_*/bbox_*/kps_*) that needs its own decoder");
    }

    const float* loc_v = loc.ptr<float>(0);
    const float* conf_v = conf.ptr<float>(0);
    const float* iou_v = iou.ptr<float>(0);
    for (auto i = 0; i < priors.size(); ++i) {
        // get score
        float cls_score = conf_v[i*2+1];
        float iou_score = iou_v[i];
        // clamp
        if (iou_score < 0.f) {
            iou_score = 0.f;
        }
        else if (iou_score > 1.f) {
            iou_score = 1.f;
        }
        float score = std::sqrt(cls_score * iou_score);

        // ignore low scores —— 注意这里必须是 `>=` 而不是 `<`：
        // NaN 与两个比较都是 false，`score < ignore_score` 会把 NaN 分数放行，
        // 于是模型输出里的未初始化尾部（见下）变成"分数=nan 的幽灵脸"。
        if (!(score >= ignore_score)) { continue; }
        // 分数有限还不够：尾部 anchor 的 loc/iou 同样是垃圾，几何必须是有限值。
        if (!std::isfinite(loc_v[i*14+0]) || !std::isfinite(loc_v[i*14+1]) ||
            !std::isfinite(loc_v[i*14+2]) || !std::isfinite(loc_v[i*14+3])) { continue; }

        Face face;
        face.score = score;
        // get bounding box
        float cx = priors[i].center.x + loc_v[i*14+0] * variance[0] * priors[i].w;
        float cy = priors[i].center.y + loc_v[i*14+1] * variance[0] * priors[i].h;
        float w  = priors[i].w * exp(loc_v[i*14+2] * variance[0]);
        float h  = priors[i].h * exp(loc_v[i*14+3] * variance[1]);
        float x1 = (cx - w / 2) * out_w;
        float y1 = (cy - h / 2) * out_h;
        float x2 = (cx + w / 2) * out_w;
        float y2 = (cy + h / 2) * out_h;
        face.bbox = { {x1, y1}, {x2, y2} };

        // 几何门：一个"检测"首先必须是**这张图里的一个框**。
        // fused 导出（yunet_120x160）最后一级 anchor 的值依赖 Net 实例的缓冲状态
        // （实测同一张图：不同实例给出 NaN / (1,0) / 看似正常的分数），分数本身
        // 被 clamp 到 [0,1] 所以无法凭分数判别 —— 但垃圾 loc 一定产生病态框。
        // 边缘被裁掉的脸仍然保留（只要求与画面有实质交集）。
        {
            const float bx1 = std::min(x1, x2), bx2 = std::max(x1, x2);
            const float by1 = std::min(y1, y2), by2 = std::max(y1, y2);
            const float bw = bx2 - bx1, bh = by2 - by1;
            if (!(bw >= 2.f && bh >= 2.f && bw <= 8.f * out_w && bh <= 8.f * out_h)) { continue; }
            const float ix = std::max(0.f, std::min(bx2, static_cast<float>(out_w)) -
                                          std::max(bx1, 0.f));
            const float iy = std::max(0.f, std::min(by2, static_cast<float>(out_h)) -
                                          std::max(by1, 0.f));
            if (ix * iy < 0.01f * bw * bh) { continue; }
        }

        // YuNet 关键点顺序: loc->[right_eye, left_eye, nose, mouth_left, mouth_right]
        // 注意 8,9 是鼻子而非嘴（历史版本在此处把 nose 与 mouth_left 接反了）
        float x_re = (priors[i].center.x + loc_v[i*14+ 4] * variance[0] * priors[i].w) * out_w;
        float y_re = (priors[i].center.y + loc_v[i*14+ 5] * variance[0] * priors[i].h) * out_h;
        float x_le = (priors[i].center.x + loc_v[i*14+ 6] * variance[0] * priors[i].w) * out_w;
        float y_le = (priors[i].center.y + loc_v[i*14+ 7] * variance[0] * priors[i].h) * out_h;
        float x_n  = (priors[i].center.x + loc_v[i*14+ 8] * variance[0] * priors[i].w) * out_w;
        float y_n  = (priors[i].center.y + loc_v[i*14+ 9] * variance[0] * priors[i].h) * out_h;
        float x_ml = (priors[i].center.x + loc_v[i*14+10] * variance[0] * priors[i].w) * out_w;
        float y_ml = (priors[i].center.y + loc_v[i*14+11] * variance[0] * priors[i].h) * out_h;
        float x_mr = (priors[i].center.x + loc_v[i*14+12] * variance[0] * priors[i].w) * out_w;
        float y_mr = (priors[i].center.y + loc_v[i*14+13] * variance[0] * priors[i].h) * out_h;
        face.landmarks = {
            {x_re, y_re}, // right eye  (人物的右眼，位于图像左侧)
            {x_le, y_le}, // left eye   (人物的左眼，位于图像右侧)
            {x_ml, y_ml}, // mouth left (人物的左嘴角，位于图像右侧)
            {x_n,  y_n }, // nose
            {x_mr, y_mr}  // mouth right
        };

        dets.push_back(face);
    }

    return dets;
}

}  // namespace facerec
