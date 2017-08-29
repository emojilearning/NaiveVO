#include <nvo_util.h>
#include <vector>
using namespace cv;
using namespace std;
namespace nvo {
    void quatFromAngularVelocity(Mat &qwt, const Mat &w) {
        qwt = Mat(4, 1, CV_64F);
        cout << w.type() << endl;
        const float x = w.at<double>(0);
        const float y = w.at<double>(1);
        const float z = w.at<double>(2);
        const float angle = sqrt(x * x + y * y + z * z);  //module of angular velocity

        if (angle > 0.0) //the formulas from the link
        {
            qwt.at<double>(0) = x * sin(angle / 2.0f) / angle;
            qwt.at<double>(1) = y * sin(angle / 2.0f) / angle;
            qwt.at<double>(2) = z * sin(angle / 2.0f) / angle;
            qwt.at<double>(3) = cos(angle / 2.0f);
        } else    //to avoid illegal expressions
        {
            qwt.at<double>(0) = qwt.at<double>(0) = qwt.at<double>(0) = 0.0f;
            qwt.at<double>(3) = 1.0f;
        }
    }

    void featureMatchByKnn(Mat &desp_m, Mat &desp_i, vector <DMatch> &matches) {

        //FlannBasedMatcher matcher2(new flann::LshIndexParams(20, 10, 2));
        if (desp_m.rows == 0 || desp_i.rows == 0)
            return;
        auto cross_matcher = BFMatcher::create(NORM_HAMMING, true);
        auto knn_matcher = BFMatcher::create(NORM_HAMMING);
        //auto knn_matcher = &matcher2;
        vector <DMatch> cross_matches;
        vector <vector<DMatch>> knn_matches;
        cross_matcher->match(desp_m, desp_i, cross_matches);
        knn_matcher->knnMatch(desp_m, desp_i, knn_matches, 2);
        vector<bool> mask(knn_matches.size(), true);
        int i = 0;
        for (auto m = knn_matches.begin(); m != knn_matches.end(); m++) {
            if (m->size() == 1)
                continue;
            else if (m->size() == 0)
                mask[i] = false;
            else if ((*m)[0].distance / (*m)[1].distance > 1.0 / 1.5)
                mask[i] = false;
            i++;
        }
        i = 0;
        for (auto m : knn_matches) {
            if (!mask[i++])
                continue;
            for (auto n : cross_matches) {
                if (m[0].trainIdx == n.trainIdx && m[0].queryIdx == n.queryIdx)
                    matches.push_back(n);
            }
        }
    }
}
