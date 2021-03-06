//
// Created by flamming on 31/05/2017.
//

#include "Camera.h"
using namespace cv;
namespace nvo
{
	Camera::Camera(double fx_, double fy_, double cx_, double cy_) :fx(fx_), fy(fy_), cx(cx_), cy(cy_)
	{
		K << fx, 0, cx,
			0, fy, cy,
			0, 0, 1;
		K_inv = K.inv();
	}
	const Matx33d & Camera::getK()
	{
		return K;
	}
	Vec2d Camera::camera2pixel(const Vec3d &p_c)
	{
		Vec3d t = K*p_c;
		return  Vec2d{ t[0] / t[2],t[1] / t[2] };
	}
	Vec3d Camera::pixel2camera(const Vec2d& p_p, double depth)
	{
		return K_inv*Vec3d(p_p[0], p_p[1], 1)*depth;
	}
}