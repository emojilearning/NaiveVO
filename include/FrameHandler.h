//
//created on 2017/8/28 by guanyi
//
#ifndef NAIVEVOFRAMEHANDLER_H
#define NAIVEVOFRAMEHANDLER_H
#include <vector>
#include <FrameStructure.h>
namespace nvo {
	class FrameHandler {
	public:
		enum Stage {
			STOPPED,
			PAUSED,
			RELOCALIZING,
			STARTED,
			OK,
			FIRST_FRAME,
		};
		FrameHandler() {};
		void solvePose(std::vector<cv::KeyPoint> &kpts_0, std::vector<cv::KeyPoint> &kpts_1, std::vector<cv::DMatch>& matches);
		void featureMatchByKnn(FramePtr last_frame, FramePtr current_frame, std::vector<cv::DMatch>& matches);
		void processFirstFrame(FramePtr frm);
		void processSecondFrame(FramePtr frm);
		void processCommonFrame(FramePtr frm);
		bool pocessFrame(FramePtr frm);
		void start();
		void solvePoseByPnP(std::vector<cv::KeyPoint> &kpts_0, std::vector<cv::KeyPoint> &kpts_1, std::vector<cv::DMatch>& matches);
		cv::Mat triangluate(std::vector<cv::Point2f> cp0,std::vector<cv::Point2f> cp1,cv::Mat mask);

	private:
		FramePtr last_frame{ 0 }, current_frame{ 0 };
		Stage stage_{ STOPPED };
		cv::Vec3d position;
		cv::Mat rotation;
		cv::Mat T_f_w;
		CameraPtr cam_;

	};
}

#endif // !NAIVEVOFRAMEHANDLER_H
