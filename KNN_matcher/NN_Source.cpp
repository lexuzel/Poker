#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "Samples_Generator.h"				// � ����� ����� ���� ������� .h i .cpp
#include "Features_Mapper.h"

using namespace std;
using namespace cv;

string imageDirectory = "./images";			// ������ ����� ��� ����������
string datasetDirectory = "./trainset";		// ������� ����� ��� ���������� � ������ ��� Features_Mapper

string dirInfo[] = { "SPADE", "CLUB", "HEART", "DIAMOND",
		"TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT",
		"NINE", "TEN", "JACK", "QUEEN", "KING", "ACE" };

/*
K-Nearest Neighbors model
ѳ��� ����������� � ����� ��� ������ � Features_Mapper: ���� float �������, �� �������������� ������.
���������:
	samples - ������� ������ ����������� ��� ����������;
	neib - ������� ���������� �����, ��������������� � ������� findNearest(), ��� ����������� ���������;
	isSuits - ������, ���� �����, �� ���� ���� ��������� ������� (4 �����) �� ������� (13 �����).
*/
void KNN(int samples, int neib,  int isSuits) {
	int startIdx = 0;
	int endIdx = 0;
	
	if (isSuits) {
		endIdx = 4;
	}
	else {
		startIdx = 4;
		endIdx = 17;
	}

	vector<Mat> trainSet;
	vector<Mat> trainSet_responses;

	/*
	���������� �� ����� dirInfo, ������ ������ ������������ �� ��������� ���������.
	startIdx �� endIdx - �������� �� �������� ������ ����.
	��� startIdx = 0; endIdx = 4 ������ ���� dirInfo[] = { "SPADE", "CLUB", "HEART", "DIAMOND" }
	��� ����� startIdx = 4; endIdx = 17;
	*/
	for (int i = startIdx; i < endIdx; i++) {
		/*
		� ������ trainSet ����������� ������� ������ featureNumber x samples, 
		�� ���� - ����������� ������� ��������� ������.
		���������� ����������� �������� Features_Mapper::prepare_descriptors()

		� ������ trainSet_responses ����������� ������� 1 x samples (����������),
		�� ������� ������ ������� ����� �������� int, �� ���� ��������������� ������ � ����� dirInfo
		*/
		trainSet.push_back(Features_Mapper::prepare_descriptors(datasetDirectory + "/" + dirInfo[i], samples, 0));
		Mat responses = Mat::zeros(Size(1, samples), CV_32SC1);
		for (int y = 0; y < samples; y++) {
			responses.at<int>(y, 0) = i;
		}
		trainSet_responses.push_back(responses);
	}
	Mat TS, TS_responses;
	vconcat(trainSet, TS);
	vconcat(trainSet_responses, TS_responses);
	/*
	�������� �������� ������� � �� Y ������� �'������� �������� ������� trainSet �� trainSet_responses �� ��������.
	� ��������� ������� TS i TS_responses ����� ������ samples x classNumber
	*/

	Ptr<ml::KNearest> KNN = ml::KNearest::create();		// ����������� ������� ������ ���

	cout << "Start training network..." << endl;
	KNN->train(ml::TrainData::create(TS, ml::ROW_SAMPLE, TS_responses));	// ������ ��������� �� ������ � TS, �������������� ������ � TS_responses;
	cout << "Finish training network..." << endl;

	/*
	�������� ���������� ����������.
	string imgPath - ���������, �� ������ ���������� ��� ����������.
	� ��� ��������� �������� Features_Mapper::prepare_descriptors() ����������� �����������.

	int imgCount - ������� ���������, �� ������ � �������� (��������� ���� ������)
	Mat prediction - ������� ����������, �� ������ float �������� (�� ��� ������� �����). 
	���� ����� �������� �� int, ��� ����������� � ����� ������� ��� dirInfo
	*/
	int imgCount = 65;
	string imgPath = "./img_predict/";
	Mat image_desc = Features_Mapper::prepare_descriptors(imgPath, imgCount, 0);
	Mat prediction;
	KNN->findNearest(image_desc, neib, prediction);
	float u = prediction.at<float>(0, 0);

	if (isSuits) {
		cout << "KNN for SUITS prediction:" << endl;
	} else cout << "KNN for RANK prediction:" << endl;
	for (int i = 0; i < imgCount; i++) {
		cout <<  i << ".	" << dirInfo[(int)prediction.at<float>(i, 0)] << endl;
	}
	cout << endl;
}

void KNN(int samples, int neib, vector<int> classesIdx, vector<int> img_for_prediction, int useExtra) {
	vector<Mat> trainSet;
	vector<Mat> trainSet_responses;

	for (int i : classesIdx) {
		trainSet.push_back(Features_Mapper::prepare_descriptors(datasetDirectory + "/" + dirInfo[i], samples, useExtra));
		Mat responses = Mat::zeros(Size(1, samples), CV_32SC1);
		for (int y = 0; y < samples; y++) {
			responses.at<int>(y, 0) = i;
		}
		trainSet_responses.push_back(responses);
	}
	Mat TS, TS_responses;
	vconcat(trainSet, TS);
	vconcat(trainSet_responses, TS_responses);
	/*
	�������� �������� ������� � �� Y ������� �'������� �������� ������� trainSet �� trainSet_responses �� ��������.
	� ��������� ������� TS i TS_responses ����� ������ samples x classNumber
	*/

	Ptr<ml::KNearest> KNN = ml::KNearest::create();		// ����������� ������� ������ ���

	cout << "Start training network..." << endl;
	KNN->train(ml::TrainData::create(TS, ml::ROW_SAMPLE, TS_responses));	// ������ ��������� �� ������ � TS, �������������� ������ � TS_responses;
	cout << "Finish training network..." << endl;

	string imgPath = "./img_predict/";
	vector<Mat> vec;
	for (int i : img_for_prediction) {
		Mat image = imread(imgPath + to_string(i) + ".jpg", IMREAD_GRAYSCALE);
		vec.push_back(Features_Mapper::prepare_descriptors(image, useExtra));
	}
	Mat image_desc, prediction;
	vconcat(vec, image_desc);
	KNN->findNearest(image_desc, neib, prediction);

	int k = 0;
	for (int i : img_for_prediction) {
		cout << i << ".	" << dirInfo[(int)prediction.at<float>(k, 0)] << endl;
		k++;
	}
	cout << endl;

	KNN->save("../../Poker_Task_Final/Poker_Task_Final/KNN_model/" + dirInfo[classesIdx[0]] + ".xml");
}

int main() {
	
	/*
	������ sampleCount = 400 ������ ��� �������� ������ ����.
	�������� ������ ��������� (� ���� ����������) � ������� ��������� (���� �������� ����������)
	�������� ����� ����������� ��������� imgSize = 51
	
	�� ������� ����� ���������������, ���� ��������� ./trainset �������.
	*/
	//Samples_Generator SG(imageDirectory, datasetDirectory, 800, 51);

	/*
	����������� ������ ��� ��� ������.
	�-��� ������ = 40;
	�-��� ���������� ����� � ������� findNearest() = 10
	*/
	string imgPath = "./img_predict/";

	//vector<int> suits;
	//for (int i = 1; i < 17; i++) {
	//	Mat image = imread(imgPath + to_string(i) + ".jpg", IMREAD_GRAYSCALE);
	//	if (!image.data) {
	//		continue;
	//	}
	//	suits.push_back(i);
	//}

	//KNN(240, 60, { 0, 1, 2, 3 }, suits, 8);

	
	int imgCount = 68;

	vector<int> two_child_contours, one_child_contours, null_child_contours;
	for (int i = 17; i < 68; i++) {
		Mat image = imread(imgPath + to_string(i) + ".jpg", IMREAD_GRAYSCALE);
		if (!image.data) {
			continue;
		}
		Mat img_th;
		normalize(image, image, 0, 255, NORM_MINMAX);
		threshold(image, img_th, 140, 255, THRESH_BINARY_INV);
		morphologyEx(img_th, img_th, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));
		vector<vector<Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		findContours(img_th, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

		int maxArea = 0;
		int maxArea_idx = 0;
		for (int j = 0; j < contours.size(); j++) {
			vector<Point> c = contours.at(j);
			int badPoint = 0;
			for (int k = 0; k < c.size(); k++) {
				Point p = c.at(k);
				if (p.x == 0 || p.x == image.cols || p.y == 0 || p.y == image.rows) {
					badPoint = 1;
				}
			}
			if (!badPoint) {
				int cArea = contourArea(c);
				if (cArea > maxArea) {
					maxArea = cArea;
					maxArea_idx = j;
				}
			}
		}
		vector<vector<Point>> kids;
		for (int j = 0; j < hierarchy.size(); j++) {
			if (hierarchy.at(j)[3] == maxArea_idx) kids.push_back(contours.at(j));
		}
		if (kids.size() == 2) {
			two_child_contours.push_back(i);
		}
		else if (kids.size() == 1) {
			one_child_contours.push_back(i);
		}
		else null_child_contours.push_back(i);
	}

	KNN(80, 20, { 10, 13 }, two_child_contours, 16);
	KNN(200, 60, { 8, 11, 12, 14 }, one_child_contours, 16);
	KNN(400, 80, { 4, 5, 6, 7, 9, 15, 16 }, null_child_contours, 8);


return 0;
}