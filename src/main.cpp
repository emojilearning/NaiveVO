#include <opencv2/opencv.hpp>
#include <FrameStructure.h>
#include <fstream>
#include <string>
#include <FrameHandler.h>
using namespace cv;
using namespace std;
using namespace nvo;

int runFromTUM(string data_path)
{
    FrameHandler nvo_instance;
    fstream rgb_file_list;
    rgb_file_list.open(data_path + "rgb.txt");
    if(!rgb_file_list.is_open())
        return -1;
    for (int i = 0; i < 3; ++i) {
        char buf[100];
        rgb_file_list.getline(buf,100);
    }

    double timestamp;
    string file_name_buf;
    nvo_instance.start();
    while (rgb_file_list>>timestamp&&rgb_file_list>>file_name_buf)
    {
        cout<<file_name_buf<<endl;
        Mat frame_img = imread(data_path+file_name_buf);
        shared_ptr<Frame> frm = make_shared<Frame>(frame_img,timestamp);
        imshow("data",frm->initialImg);
        nvo_instance.pocessFrame(frm);
        waitKey(1);
    }
    return 0;
}

int main()
{
#ifdef WIN32
    string data_path = "../../rgbd_dataset_freiburg1_desk2/";
#else
	string data_path = "../rgbd_dataset_freiburg1_desk2/";
#endif
	if(runFromTUM(data_path)!=0)
    {
        cout<<"the path may be not correct"<<endl;
    }
    return 0;
}
