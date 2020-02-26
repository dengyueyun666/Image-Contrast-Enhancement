#include <iostream>
#include <opencv2/opencv.hpp>

#include "image_enhancement.h"

void AGCIE(const cv::Mat & src, cv::Mat & dst)
{
	int rows = src.rows;
	int cols = src.cols;
	int channels = src.channels();
	int total_pixels = rows * cols;

	cv::Mat L;
	cv::Mat HSV;
	std::vector<cv::Mat> HSV_channels;
	if (channels == 1) {
		L = src.clone();
	}
	else {
		cv::cvtColor(src, HSV, CV_BGR2HSV_FULL);
		cv::split(HSV, HSV_channels);
		L = HSV_channels[2];
	}

	cv::Mat L_norm;
	L.convertTo(L_norm, CV_64F, 1.0 / 255.0);

	cv::Mat mean, stddev;
	cv::meanStdDev(L_norm, mean, stddev);
	double mu = mean.at<double>(0, 0);
	double sigma = stddev.at<double>(0, 0);

	double tau = 3.0;

	double gamma;
	if (4 * sigma <= 1.0 / tau) { // low-contrast
		gamma = -std::log2(sigma);
	}
	else { // high-contrast
		gamma = std::exp((1.0 - mu - sigma) / 2.0);
	}
	
	std::vector<double> table_double(256, 0);
	for (int i = 1; i < 256; i++) {
		table_double[i] = i / 255.0;
	}

	if (mu >= 0.5) { // bright image
		for (int i = 1; i < 256; i++) {
			table_double[i] = std::pow(table_double[i], gamma);
		}
	}
	else { // dark image
		double mu_gamma = std::pow(mu, gamma);
		for (int i = 1; i < 256; i++) {
			double in_gamma = std::pow(table_double[i], gamma);;
			table_double[i] = in_gamma / (in_gamma + (1.0 - in_gamma) * mu_gamma);
		}
	}
	
	std::vector<uchar> table_uchar(256, 0);
	for (int i = 1; i < 256; i++) {
		table_uchar[i] = cv::saturate_cast<uchar>(255.0 * table_double[i]);
	}

	cv::LUT(L, table_uchar, L);

	if (channels == 1) {
		dst = L.clone();
	}
	else {
		cv::merge(HSV_channels, dst);
		cv::cvtColor(dst, dst, CV_HSV2BGR_FULL);
	}

	return;
}