#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "Features_Mapper.h"

using namespace std;
using namespace cv;

Features_Mapper::Features_Mapper() {}

Mat prepare_image(Mat& image, vector<Point>& contour) {
	//RotatedRect rect = minAreaRect(contour);
	//rect.size;
	//Point2f corners[4];
	//rect.points(corners);

	//auto a = norm(corners[0] - corners[1]);
	//auto b = norm(corners[1] - corners[2]);

	//vector<Point2f> c = { corners[0], corners[1], corners[2], corners[3] };

	//const static auto vectorContourToMat = [](const vector<Point2f>& contour) {
	//	Mat contour_in_mat(contour);
	//	contour_in_mat.convertTo(contour_in_mat, CV_32F);
	//	return contour_in_mat;
	//};

	//Mat paper(Size(min(a, b), max(a, b)), CV_8UC1);
	//Mat tmp1 = vectorContourToMat(c);
	//Mat tmp2;
	///*
	//A  B

	//D  C
	//*/
	//if (b > a)		// Якщо сторона 0-1 менша від 1-2, то будуємо прямокутник з точки D проти годинникової стрілки.
	//	tmp2 = static_cast<Mat>(Mat_<float>(8, 1, CV_32F) << 0, b, a, b, a, 0, 0, 0).reshape(2);
	//if (a > b)		// Якщо сторона 0-1 більша від 1-2, то будуємо прямокутник з точки С проти годинникової стрілки.
	//	tmp2 = static_cast<Mat>(Mat_<float>(8, 1, CV_32F) << b, a, b, 0, 0, 0, 0, a).reshape(2);
	//auto p_trans_mat = getPerspectiveTransform(tmp1, tmp2);		// Знаходимо матрицю перетворення
	//warpPerspective(image, paper, p_trans_mat, paper.size());
	//return paper;
}

vector<Point> Features_Mapper::prepare_contour(Mat& image)
{
	threshold(image, image, 150, 255, THRESH_BINARY);
	vector<vector<Point>> contours;
	findContours(image, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	int cArea = 0;
	int maxArea = 0;
	int maxArea_idx = 0;
	if (contours.size() != 1)
		for (int j = 0; j < contours.size(); j++) {
			vector<Point> c = contours.at(j);
			cArea = contourArea(c);
			if (cArea > maxArea) {
				maxArea = cArea;
				maxArea_idx = j;
			}
					
		}
	return contours.at(maxArea_idx);
}

Vec2d Features_Mapper::calculateCoef(vector<Point>& c)
{
	vector<Point> hull;
	convexHull(c, hull);
	return { contourArea(hull) / contourArea(c) , contourArea(c) / arcLength(c, true) };
}

vector<vector<Point>> Features_Mapper::split_contour(vector<Point>& c)
{
	int minY = 500;
	int maxY = 0;
	for (int i = 0; i < c.size(); i++) {
		Point p = c.at(i);
		if (p.y < minY) minY = p.y;
		if (p.y > maxY) maxY = p.y;
	}
	int meanY = (maxY + minY) / 2;
	vector<Point> top, bottom;
	for (int i = 0; i < c.size(); i++) {
		Point p = c.at(i);
		if (p.y < meanY) top.push_back(p);
		else bottom.push_back(p);
	}
	return { top, bottom };
}

Mat Features_Mapper::prepare_descriptors(string datasetDirectory, int n){
	Mat trainSet = Mat::zeros(Size(3, n), CV_32FC1);
	for (int i = 0; i < n; i++) {
		Mat image = imread(datasetDirectory + "/" + to_string(i) + ".jpg", IMREAD_GRAYSCALE);
		vector<Point> pattern_contour = prepare_contour(image);
		vector<vector<Point>> splited = split_contour(pattern_contour);

		RotatedRect rect = minAreaRect(pattern_contour);
		double c_coef1 = rect.size.area() / contourArea(pattern_contour);

		rect = minAreaRect(splited[0]);
		double top_coef = rect.size.area() / contourArea(splited[0]);

		rect = minAreaRect(splited[1]);
		double bottom_coef = rect.size.area() / contourArea(splited[1]);

		trainSet.at<float>(0, i) = c_coef;
		trainSet.at<float>(1, i) = top_coef;
		trainSet.at<float>(2, i) = bottom_coef;

		/*vector<Point> pattern_contour = prepare_contour(image);
		Mat object = prepare_image(image, pattern_contour);
		pattern_contour = prepare_contour(object);
		vector<vector<Point>> splited = split_contour(pattern_contour);

		double c_coef = object.cols * object.rows / contourArea(pattern_contour);
		double top_coef = contourArea(splited[0]) / arcLength(splited[0], true);
		double bottom_coef = contourArea(splited[1]) / arcLength(splited[1], true);

		*/

	}
	return trainSet;
}

Mat Features_Mapper::prepare_descriptors(Mat image){
	Mat m = Mat::zeros(Size(3, 1), CV_32FC1);
	vector<Point> pattern_contour = prepare_contour(image);
	vector<vector<Point>> splited = split_contour(pattern_contour);

	RotatedRect rect = minAreaRect(pattern_contour);
	double c_coef = rect.size.area() / contourArea(pattern_contour);

	rect = minAreaRect(splited[0]);
	double top_coef = rect.size.area() / contourArea(splited[0]);

	rect = minAreaRect(splited[1]);
	double bottom_coef = rect.size.area() / contourArea(splited[1]);


	/*vector<Point> pattern_contour = prepare_contour(image);
	Mat object = prepare_image(image, pattern_contour);
	pattern_contour = prepare_contour(object);
	vector<vector<Point>> splited = split_contour(pattern_contour);

	double c_coef = object.cols * object.rows / contourArea(pattern_contour);
	double top_coef = contourArea(splited[0]) / arcLength(splited[0], true);
	double bottom_coef = contourArea(splited[1]) / arcLength(splited[1], true);*/

	m.at<float>(0, 0) = c_coef;
	m.at<float>(1, 0) = top_coef;
	m.at<float>(2, 0) = bottom_coef;

	return m;
}

Features_Mapper::~Features_Mapper() {}
