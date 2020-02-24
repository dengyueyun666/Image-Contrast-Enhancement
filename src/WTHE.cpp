#include <iostream>
#include <opencv2/opencv.hpp>

#include "image_enhancement.h"

void WTHE(const cv::Mat & src, cv::Mat & dst, float r, float v)
{
	int rows = src.rows;
	int cols = src.cols;
	int channels = src.channels();
	int total_pixels = rows * cols;

	cv::Mat L;
	cv::Mat YUV;
	std::vector<cv::Mat> YUV_channels;
	if (channels == 1) {
		L = src.clone();
	}
	else {
		cv::cvtColor(src, YUV, CV_BGR2YUV);
		cv::split(YUV, YUV_channels);
		L = YUV_channels[0];
	}

	int histsize = 256;
	float range[] = { 0,256 };
	const float* histRanges = { range };
	int bins = 256;
	cv::Mat hist;
	calcHist(&L, 1, 0, cv::Mat(), hist, 1, &histsize, &histRanges, true, false);

	float total_pixels_inv = 1.0f / total_pixels;
	cv::Mat P = hist.clone();
	for (int i = 0; i < 256; i++) {
		P.at<float>(i) = P.at<float>(i) * total_pixels_inv;
	}

	cv::Mat Pwt = P.clone();
	double minP, maxP;
	cv::minMaxLoc(P, &minP, &maxP);
	float Pu = v * maxP;
	float Pl = minP;
	for (int i = 0; i < 256; i++) {
		float Pi = P.at<float>(i);
		if (Pi > Pu)
			Pwt.at<float>(i) = Pu;
		else if (Pi < Pl)
			Pwt.at<float>(i) = 0;
		else
			Pwt.at<float>(i) = std::pow((Pi - Pl) / (Pu - Pl), r) * Pu;
	}

	cv::Mat Cwt = Pwt.clone();
	float cdf = 0;
	for (int i = 0; i < 256; i++) {
		cdf += Pwt.at<float>(i);
		Cwt.at<float>(i) = cdf;
	}

	float Wout = 255.0f;
	float Madj = 0.0f;
	std::vector<uchar> table(256, 0);
	for (int i = 0; i < 256; i++) {
		table[i] = cv::saturate_cast<uchar>(Wout * Cwt.at<float>(i) + Madj);
	}

	cv::LUT(L, table, L);

	if (channels == 1) {
		dst = L.clone();
	}
	else {
		cv::merge(YUV_channels, dst);
		cv::cvtColor(dst, dst, CV_YUV2BGR);
	}

	return;
}