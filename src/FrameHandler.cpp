#include <FrameHandler.h>
#include <opencv2/opencv.hpp>
#include <nvo_util.h>
using namespace cv;
using namespace std;

namespace nvo {

	void FrameHandler::processFirstFrame(FramePtr frm)
	{
		current_frame = frm;
		stage_ = FIRST_FRAME;
		//current_frame->t = position = { 0,0,0 };
  //      current_frame->R = rotation = Mat::eye(3,3,CV_64F);
        T_f_w = Mat::eye(4,4,CV_64F);
		current_frame->T_f_w = T_f_w.clone();
	}





	Mat FrameHandler::triangluate(vector<Point2f> cp0,vector<Point2f> cp1,Mat mask)
	{
        Mat proj1 = Mat(cam_->getK()) * last_frame->T_f_w(Range(0,3),Range(0,4));
		Mat proj2 = Mat(cam_->getK()) * current_frame->T_f_w(Range(0, 3), Range(0, 4));
        Mat outp;
		//cout << proj1 << endl;;
		//cout << proj2 << endl;
		triangulatePoints(proj1, proj2, cp0, cp1, outp);
        //cout<< proj1 <<endl;
        //cout<<outp.type()<<endl;
		for (int i = 0; i<cp0.size(); i++)
		{
			if (mask.at<char>(i))
			{
				auto op = outp.col(i);
				auto w = op.at<float>(3);
				Point3f p3d = { op.at<float>(0) / w ,op.at<float>(1) / w,op.at<float>(2) / w };
//				last_frame->m_p_op[cp0[i]] = { op.at<float>(0) / w ,op.at<float>(1) / w,op.at<float>(2) / w };
				if (last_frame->m_p_op.find(cp0[i])!=last_frame->m_p_op.end())
				{
//					cout << p3d << "----------" << last_frame->m_p_op[cp0[i]] << endl;
				}
				current_frame->m_p_op[cp1[i]] = p3d;
			}
		}
        return outp;
	}

	void FrameHandler::solvePose(vector<KeyPoint> &kpts_0, vector<KeyPoint> &kpts_1, vector<DMatch>& matches)
	{
		vector<Point2f> cp0;
		vector<Point2f> cp1;
		for (auto correspondence : matches)
		{
			cp0.push_back(kpts_0[correspondence.queryIdx].pt);
			cp1.push_back(kpts_1[correspondence.trainIdx].pt);
		}
		Mat mask;
		Mat E = findEssentialMat(cp0, cp1,cam_->getK(), FM_RANSAC, 0.999,1.0,mask);

        Mat out2;
        drawMatches(last_frame->initialImg, last_frame->kpts, current_frame->initialImg, current_frame->kpts , matches, out2,Scalar::all(-1),Scalar::all(-1),mask);
        imshow("matches", out2);

		Mat R;
		Vec3d t;
		recoverPose(E, cp0, cp1, cam_->getK(), R, t);
        Mat T = Mat::eye(4,4,CV_64F);
        R.convertTo(T(Range(0,3),Range(0,3)),CV_64F);
        T.at<double>(0,3) = t[0];
        T.at<double>(1,3) = t[1];
        T.at<double>(2,3) = t[2];


        //current_frame->R = rotation = R*rotation;
        //current_frame->t = (position = position + Vec3d(t));
		T_f_w = T*T_f_w;
		current_frame->T_f_w = T_f_w.clone();
        cout << T_f_w.col(3).t() << endl;
        Mat tempRout;
        Rodrigues(T(Range(0, 3), Range(0, 3)),tempRout);
        Mat Rout;
        cout<<"asdasd"<<endl;
        quatFromAngularVelocity(Rout,tempRout);
        cout <<Rout.t()<<endl;

        Mat outp = triangluate(cp0,cp1,mask);
	}

	void FrameHandler::processSecondFrame(FramePtr frm)
	{
		last_frame = current_frame;
		current_frame = frm;
		vector<DMatch> matches;
		featureMatchByKnn(last_frame->descriptor, current_frame->descriptor, matches);
		solvePose(last_frame->kpts, current_frame->kpts, matches);
		stage_ = OK;
	}


	void FrameHandler::solvePoseByPnP(vector<KeyPoint> &kpts_0, vector<KeyPoint> &kpts_1, vector<DMatch>& matches)
	{
		vector<Point3f> cp0;
		vector<Point2f> cp1;

        vector<Point2f> ccp0;
        vector<Point2f> ccp1;
		for (auto correspondence : matches)
		{
            ccp0.push_back(kpts_0[correspondence.queryIdx].pt);
            ccp1.push_back(kpts_1[correspondence.trainIdx].pt);

            if (last_frame->m_p_op.find(kpts_0[correspondence.queryIdx].pt) != last_frame->m_p_op.end())
			{
				cp0.push_back(last_frame->m_p_op[kpts_0[correspondence.queryIdx].pt]);
				cp1.push_back(kpts_1[correspondence.trainIdx].pt);
			}
		}
		vector<int> mask;
		Mat RR;
		Mat t;
        cout << cp0.size() << endl;

        solvePnPRansac(cp0, cp1, cam_->getK(), noArray(), RR, t,false,100,8,0.999,mask);
        Mat R;
        Rodrigues(RR,R);
		Mat T = Mat::eye(4, 4, CV_64F);
		R.convertTo(T(Range(0, 3), Range(0, 3)), CV_64F);
		T.at<double>(0, 3) = t.at<double>(0);
		T.at<double>(1, 3) = t.at<double>(1);
		T.at<double>(2, 3) = t.at<double>(2);
		T_f_w = T*T_f_w;
		current_frame->T_f_w = T_f_w.clone();
        Mat tempRout;
        Rodrigues(T(Range(0, 3), Range(0, 3)),tempRout);
		//cout << T_f_w << endl;
		cout << T_f_w.col(3).t() << endl;

        Mat Rout;
        quatFromAngularVelocity(Rout,tempRout);
        cout <<Rout.t()<<endl;
		for (int i = 0; i<mask.size(); i++)
		{
			current_frame->m_p_op[cp1[mask[i]]] = cp0[mask[i]];
		}
        Mat mask1;
        Mat E = findEssentialMat(ccp0, ccp1,cam_->getK(), FM_RANSAC, 0.999,1.0,mask1);
        triangluate(ccp0,ccp1,mask1);

        Mat out2;
        drawMatches(last_frame->initialImg, last_frame->kpts, current_frame->initialImg, current_frame->kpts , matches, out2,Scalar::all(-1),Scalar::all(-1),mask1);
        imshow("matches", out2);

	}
	void FrameHandler::processCommonFrame(FramePtr frm)
	{
		last_frame = current_frame;
		current_frame = frm;
		vector<DMatch> matches;
		featureMatchByKnn(last_frame->descriptor, current_frame->descriptor, matches);
		solvePoseByPnP(last_frame->kpts, current_frame->kpts, matches);
	}

	bool FrameHandler::pocessFrame(FramePtr frm) {
		frm->extractFeature();
		switch (stage_)
		{
		case STARTED:
			processFirstFrame(frm);
			break;
		case FIRST_FRAME:
			processSecondFrame(frm);
			break;
		case OK:
			processCommonFrame(frm);
			break;
		default:
			break;

		}
		return true;
	};

	void FrameHandler::start() {
		double f = 525.0;
		double cx = 319.5;
		double cy = 239.5;
		stage_ = STARTED;
		cam_ = make_shared<Camera>(f, f, cx, cy);
	}



}