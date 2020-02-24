// This must be defnined, in order to use arma::spsolve in the code with SuperLU
#define ARMA_USE_SUPERLU

#include <armadillo>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "util.h"

arma::sp_mat spdiags(const arma::mat& B, const std::vector<int>& d, int m, int n)
{
    arma::sp_mat A(m, n);
    for (int k = 0; k < d.size(); k++) {
        int i_min = std::max(0, -d[k]);
        int i_max = std::min(m - 1, n - d[k] - 1);
        A.diag(d[k]) = B(arma::span(0, i_max - i_min), arma::span(k, k));
    }

    return A;
}


// This is a OpenCV-based implementation of conv2 in Matlab.
cv::Mat conv2(const cv::Mat &img, const cv::Mat& ikernel, ConvolutionType type)
{
	cv::Mat dest;
	cv::Mat kernel;
	cv::flip(ikernel, kernel, -1);
	cv::Mat source = img;
	if (CONVOLUTION_FULL == type)
	{
		source = cv::Mat();
		const int additionalRows = kernel.rows - 1, additionalCols = kernel.cols - 1;
		copyMakeBorder(img, source, (additionalRows + 1) / 2, additionalRows / 2, (additionalCols + 1) / 2, additionalCols / 2, cv::BORDER_CONSTANT, cv::Scalar(0));
	}
	cv::Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);
	int borderMode = cv::BORDER_CONSTANT;
	filter2D(source, dest, img.depth(), kernel, anchor, 0, borderMode);

	if (CONVOLUTION_VALID == type)
	{
		dest = dest.colRange((kernel.cols - 1) / 2, dest.cols - kernel.cols / 2).rowRange((kernel.rows - 1) / 2, dest.rows - kernel.rows / 2);
	}
	return dest;
}