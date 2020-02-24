#include <iostream>
#include <opencv2/opencv.hpp>

#include "image_enhancement.h"

void JHE(const cv::Mat & src, cv::Mat & dst)
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

	// Compute average image.
	cv::Mat avg_L;
	cv::boxFilter(L, avg_L, -1, cv::Size(3, 3), cv::Point(-1,-1), true, cv::BORDER_CONSTANT);

	// Computer joint histogram.
	cv::Mat jointHist = cv::Mat::zeros(256, 256, CV_32S);
	for (int r = 0; r < rows; r++) {
		uchar* L_it = L.ptr<uchar>(r);
		uchar* avg_L_it = avg_L.ptr<uchar>(r);
		for (int c = 0; c < cols; c++) {
			int i = L_it[c];
			int j = avg_L_it[c];
			jointHist.at<int>(i, j)++;
		}
	}

	// Compute CDF.
	cv::Mat CDF = cv::Mat::zeros(256, 256, CV_32S);
	int min_CDF = total_pixels + 1;
	int cumulative = 0;
	for (int i = 0; i < 256; i++) {
		int* jointHist_it = jointHist.ptr<int>(i);
		int* CDF_it = CDF.ptr<int>(i);
		for (int j = 0; j < 256; j++) {
			int count = jointHist_it[j];
			cumulative += count;
			if (cumulative > 0 && cumulative < min_CDF)
				min_CDF = cumulative;
			CDF_it[j] = cumulative;
		}
	}

	// Compute equalized joint histogram.
	cv::Mat h_eq = cv::Mat::zeros(256, 256, CV_8U);
	for (int i = 0; i < 256; i++) {
		uchar* h_eq_it = h_eq.ptr<uchar>(i);
		int* cdf_it = CDF.ptr<int>(i);
		for (int j = 0; j < 256; j++) {
			int cur_cdf = cdf_it[j];
			h_eq_it[j] = cv::saturate_cast<uchar>(255.0 * (cur_cdf - min_CDF) / (total_pixels - 1));
		}
	}

	// Map to get enhanced image.
	for (int r = 0; r < rows; r++) {
		uchar* L_it = L.ptr<uchar>(r);
		uchar* avg_L_it = avg_L.ptr<uchar>(r);
		for (int c = 0; c < cols; c++) {
			int i = L_it[c];
			int j = avg_L_it[c];
			L_it[c] = h_eq.at<uchar>(i, j);
		}
	}

	if (channels == 1) {
		dst = L.clone();
	}
	else {
		cv::merge(YUV_channels, dst);
		cv::cvtColor(dst, dst, CV_YUV2BGR);
	}

	return;
}