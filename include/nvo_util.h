#ifndef NVO_UTIL_H
#define NVO_UTIL_H

#include <opencv2/opencv.hpp>
namespace nvo {
    void quatFromAngularVelocity(cv::Mat &qwt, const cv::Mat &w);

    void featureMatchByKnn(cv::Mat &desp_m, cv::Mat &desp_i, std::vector<cv::DMatch> &matches);
}
#endif