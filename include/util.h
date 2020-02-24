#ifndef _UTIL_H
#define _UTIL_H

// This must be defnined, in order to use arma::spsolve in the code with SuperLU
#define ARMA_USE_SUPERLU

#include <armadillo>
#include <iostream>
#include <opencv2/opencv.hpp>

// This is a Armadillo-based implementation of spdiags in Matlab.
arma::sp_mat spdiags(const arma::mat& B, const std::vector<int>& d, int m, int n);


enum ConvolutionType {
	/* Return the full convolution, including border */
	CONVOLUTION_FULL,
	/* Return only the part that corresponds to the original image */
	CONVOLUTION_SAME,
	/* Return only the submatrix containing elements that were not influenced by the border */
	CONVOLUTION_VALID
};

// This is a OpenCV-based implementation of conv2 in Matlab.
cv::Mat conv2(const cv::Mat &img, const cv::Mat& ikernel, ConvolutionType type);

#endif