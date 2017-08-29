//
// Created by flamming on 01/07/2017.
//

#include <opencv2/opencv.hpp>
#include <memory>
#include <Camera.h>
#include <unordered_map>

#ifndef NVO_FRAMESTRUCTURE_H
#define NVO_FRAMESTRUCTURE_H
namespace nvo
{
    cv::Mat convertToGray(cv::Mat& img);
    class Frame
    {
    public:
        Frame(cv::Mat img, double ts = 0) :initialImg(img),image(convertToGray(img)),timestamp(ts){};
        void extractFeature(int n_features = 2000);
        void extractFeature(int n_feature, cv::Mat mask);
        void init(int n_faeture);
        cv::Mat image;
        cv::Mat initialImg;
        std::vector<cv::KeyPoint> kpts;
        cv::Mat descriptor;
        double timestamp;
        cv::Mat R;
        cv::Vec3d t;
        cv::Mat E;
		struct point_hash
		{
			size_t operator()(cv::Point2i p) const { return p.x + p.y * 10000; }
		};
		std::unordered_map < cv::Point2i, cv::Point3f, point_hash > m_p_op;
    };
    typedef std::shared_ptr<Frame> FramePtr;

}
#endif // !NVO_FRAMESTRUCTURE_H
