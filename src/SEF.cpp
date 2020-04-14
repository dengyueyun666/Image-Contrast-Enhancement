#include <iostream>
#include <opencv2/opencv.hpp>

#include "image_enhancement.h"

std::vector<cv::Mat> gaussian_pyramid(const cv::Mat& src, int nLevel)
{
	cv::Mat I = src.clone();
	std::vector<cv::Mat> pyr;
	pyr.push_back(I);
	for (int i = 2; i <= nLevel; i++) {
		cv::pyrDown(I, I);
		pyr.push_back(I);
	}
	return pyr;
}

std::vector<cv::Mat> laplacian_pyramid(const cv::Mat& src, int nLevel)
{
	cv::Mat I = src.clone();
	std::vector<cv::Mat> pyr;
	cv::Mat J = I.clone();
	for (int i = 1; i < nLevel; i++) {
		cv::pyrDown(J, I);
		cv::Mat J_up;
		cv::pyrUp(I, J_up, J.size());
		pyr.push_back(J - J_up);
		J = I;
	}
	pyr.push_back(J); // the coarest level contains the residual low pass image
	return pyr;
}

cv::Mat reconstruct_laplacian_pyramid(const std::vector<cv::Mat>& pyr)
{
	int nLevel = pyr.size();
	cv::Mat R = pyr[nLevel - 1].clone();
	for (int i = nLevel - 2; i >= 0; i--) {
		cv::pyrUp(R, R, pyr[i].size());
		R = pyr[i] + R;
	}
	return R;
}

cv::Mat multiscale_blending(const std::vector<cv::Mat>& seq, const std::vector<cv::Mat>& W)
{
	int h = seq[0].rows;
	int w = seq[0].cols;
	int n = seq.size();

	int nScRef = int(std::log(std::min(h, w)) / log(2));

	int nScales = 1;
	int hp = h;
	int wp = w;
	while(nScales < nScRef) {
		nScales++;
		hp = (hp + 1) / 2;
		wp = (wp + 1) / 2;
	}
	//std::cout << "Number of scales: " << nScales << ", residual's size: " << hp << " x " << wp << std::endl;

	std::vector<cv::Mat> pyr;
	hp = h;
	wp = w;
	for (int scale = 1; scale <= nScales; scale++) {
		pyr.push_back(cv::Mat::zeros(hp, wp, CV_64F));
		hp = (hp + 1) / 2;
		wp = (wp + 1) / 2;
	}

	for (int i = 0; i < n; i++) {
		std::vector<cv::Mat> pyrW = gaussian_pyramid(W[i], nScales);
		std::vector<cv::Mat> pyrI = laplacian_pyramid(seq[i], nScales);

		for (int scale = 0; scale < nScales; scale++) {
			pyr[scale] += pyrW[scale].mul(pyrI[scale]);
		}
	}

	return reconstruct_laplacian_pyramid(pyr);
}

void robust_normalization(const cv::Mat& src, cv::Mat& dst, double wSat = 1.0, double bSat = 1.0)
{
	int H = src.rows;
	int W = src.cols;
	int D = src.channels();
	int N = H * W;

	double vmax;
	double vmin;
	if (D > 1) {
		std::vector<cv::Mat> src_channels;
		cv::split(src, src_channels);

		cv::Mat max_channel;
		cv::max(src_channels[0], src_channels[1], max_channel);
		cv::max(max_channel, src_channels[2], max_channel);
		cv::Mat max_channel_sort;
		cv::sort(max_channel.reshape(1,1), max_channel_sort, CV_SORT_ASCENDING);
		vmax = max_channel_sort.at<double>(int(N - wSat*N / 100 + 1));

		cv::Mat min_channel;
		cv::min(src_channels[0], src_channels[1], min_channel);
		cv::min(min_channel, src_channels[2], min_channel);
		cv::Mat min_channel_sort;
		cv::sort(min_channel.reshape(1, 1), min_channel_sort, CV_SORT_ASCENDING);
		vmin = min_channel_sort.at<double>(int(bSat*N / 100));
	}
	else {
		cv::Mat src_sort;
		cv::sort(src.reshape(1, 1), src_sort, CV_SORT_ASCENDING);
		vmax = src_sort.at<double>(int(N - wSat*N / 100 + 1));
		vmin = src_sort.at<double>(int(bSat*N / 100));
	}

	if (vmax <= vmin) {
		if (D > 1)
			dst = cv::Mat(H, W, src.type(), cv::Scalar(vmax, vmax, vmax));
		else 
			dst = cv::Mat(H, W, src.type(), cv::Scalar(vmax));
	}
	else {
		cv::Scalar Ones;
		if (D > 1) {
			cv::Mat vmin3 = cv::Mat(H, W, src.type(), cv::Scalar(vmin, vmin, vmin));
			cv::Mat vmax3 = cv::Mat(H, W, src.type(), cv::Scalar(vmax, vmax, vmax));
			dst = (src - vmin3).mul(1.0 / (vmax3 - vmin3));
			Ones = cv::Scalar(1.0, 1.0, 1.0);
		}
		else {
			dst = (src - vmin) / (vmax - vmin);
			Ones = cv::Scalar(1.0);
		}
		
		cv::Mat mask_over = dst > vmax;
		cv::Mat mask_below = dst < vmin;
		mask_over.convertTo(mask_over, CV_64F, 1.0 / 255.0);
		mask_below.convertTo(mask_below, CV_64F, 1.0 / 255.0);
		
		dst = dst.mul(Ones - mask_over) + mask_over;
		dst = dst.mul(Ones - mask_below);
	}

	return;
}

/***
@inproceedings{hessel2020extended,
  title={An extended exposure fusion and its application to single image contrast enhancement},
  author={Hessel, Charles and Morel, Jean-Michel},
  booktitle={The IEEE Winter Conference on Applications of Computer Vision},
  pages={137--146},
  year={2020}
}

This is a reimplementation from https://github.com/chlsl/simulated-exposure-fusion-ipol/
***/
void SEF(const cv::Mat & src, cv::Mat & dst, double alpha, double beta, double lambda)
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

	cv::Mat src_norm;
	src.convertTo(src_norm, CV_64F, 1.0 / 255.0);

	cv::Mat C;
	if (channels == 1) {
		C = src_norm.mul(1.0 / (L_norm + std::pow(2, -16)));
	}
	else {
		cv::Mat temp = 1.0 / (L_norm + std::pow(2, -16));
		std::vector<cv::Mat> temp_arr = { temp.clone(),temp.clone(),temp.clone() };
		cv::Mat temp3;
		cv::merge(temp_arr, temp3);
		C = src_norm.mul(temp3);
	}
	
	// Compute median
	cv::Mat tmp = src.reshape(1, 1);
	cv::Mat sorted;
	cv::sort(tmp, sorted, CV_SORT_ASCENDING);
	double med = double(sorted.at<uchar>(rows * cols * channels / 2)) / 255.0;
	//std::cout << "med = " << med << std::endl;

	//Compute optimal number of images
	int Mp = 1;					// Mp = M - 1; M is the total number of images
	int Ns = int(Mp * med);		// number of images generated with fs
	int N = Mp - Ns;			// number of images generated with f
	int Nx = std::max(N, Ns);	// used to compute maximal factor
	double tmax1 = (1.0 + (Ns + 1.0) * (beta - 1.0) / Mp) / (std::pow(alpha, 1.0 / Nx));			// t_max k=+1
	double tmin1s = (-beta + (Ns - 1.0) * (beta - 1.0) / Mp) / (std::pow(alpha, 1.0 / Nx)) + 1.0;	// t_min k=-1
	double tmax0 = 1.0 + Ns*(beta - 1.0) / Mp;														// t_max k=0
	double tmin0 = 1.0 - beta + Ns*(beta - 1.0) / Mp;												// t_min k=0
	while (tmax1 < tmin0 || tmax0 < tmin1s) {
		Mp++;
		Ns = int(Mp * med);
		N = Mp - Ns;
		Nx = std::max(N, Ns);
		tmax1 = (1.0 + (Ns + 1.0) * (beta - 1.0) / Mp) / (std::pow(alpha, 1.0 / Nx));
		tmin1s = (-beta + (Ns - 1.0) * (beta - 1.0) / Mp) / (std::pow(alpha, 1.0 / Nx)) + 1.0;
		tmax0 = 1.0 + Ns*(beta - 1.0) / Mp;
		tmin0 = 1.0 - beta + Ns*(beta - 1.0) / Mp;
		if (Mp > 49) {
			std::cerr << "The estimation of the number of image required in the sequence stopped, please check the parameters!" << std::endl;
		}
	}

	// std::cout << "M = " << Mp + 1 << ", with N = " << N << " and Ns = " << Ns << std::endl;

	// Remapping functions
	auto fun_f = [alpha, Nx](cv::Mat t, int k) {	// enhance dark parts
		return std::pow(alpha, k * 1.0 / Nx) * t;
	};
	auto fun_fs = [alpha, Nx](cv::Mat t, int k) {	// enhance bright parts
		return std::pow(alpha, -k * 1.0 / Nx) * (t - 1.0) + 1.0;
	};

	// Offset for the dynamic range reduction (function "g")
	auto fun_r = [beta, Ns, Mp](int k) {
		return (1.0 - beta / 2.0) - (k + Ns) * (1.0 - beta) / Mp;
	};

	// Reduce dynamic (using offset function "r")
	double a = beta / 2 + lambda;
	double b = beta / 2 - lambda;
	auto fun_g = [fun_r, beta, a, b, lambda](cv::Mat t, int k) {
		auto rk = fun_r(k);
		cv::Mat diff = t - rk;
		cv::Mat abs_diff = cv::abs(diff);

		cv::Mat mask = abs_diff <= beta / 2;
		mask.convertTo(mask, CV_64F, 1.0 / 255.0);

		cv::Mat sign = diff.mul(1.0 / abs_diff);

		return mask.mul(t) + (1.0 - mask).mul(sign.mul(a - lambda * lambda / (abs_diff - b)) + rk);
	};

	// final remapping functions: h = g o f
	auto fun_h = [fun_f, fun_g](cv::Mat t, int k) {		// create brighter images (k>=0) (enhance dark parts)
		return fun_g(fun_f(t, k), k);
	};
	auto fun_hs = [fun_fs, fun_g](cv::Mat t, int k) {	// create darker images (k<0) (enhance bright parts)
		return fun_g(fun_fs(t, k), k);
	};

	// derivative of g with respect to t
	auto fun_dg = [fun_r, beta, b, lambda](cv::Mat t, int k) {
		auto rk = fun_r(k);
		cv::Mat diff = t - rk;
		cv::Mat abs_diff = cv::abs(diff);

		cv::Mat mask = abs_diff <= beta / 2;
		mask.convertTo(mask, CV_64F, 1.0 / 255.0);

		cv::Mat p;
		cv::pow(abs_diff - b, 2, p);

		return mask + (1.0 - mask).mul(lambda * lambda / p);
	};

	// derivative of the remapping functions: dh = f' x g' o f
	auto fun_dh = [alpha, Nx, fun_f, fun_dg](cv::Mat t, int k) {
		return std::pow(alpha, k * 1.0 / Nx) * fun_dg(fun_f(t, k), k);
	};
	auto fun_dhs = [alpha, Nx, fun_fs, fun_dg](cv::Mat t, int k) {
		return std::pow(alpha, -k * 1.0 / Nx) * fun_dg(fun_fs(t, k), k);
	};

	// Simulate a sequence from image L_norm and compute the contrast weights
	std::vector<cv::Mat> seq(N + Ns + 1);
	std::vector<cv::Mat> wc(N + Ns + 1);

	for (int k = -Ns; k <= N; k++) {
		cv::Mat seq_temp, wc_temp;
		if (k < 0) {
			seq_temp = fun_hs(L_norm, k);	// Apply remapping function
			wc_temp = fun_dhs(L_norm, k);	// Compute contrast measure
		}
		else {
			seq_temp = fun_h(L_norm, k);	// Apply remapping function
			wc_temp = fun_dh(L_norm, k);	// Compute contrast measure
		}

		// Detect values outside [0,1]
		cv::Mat mask_sup = seq_temp > 1.0;
		cv::Mat mask_inf = seq_temp < 0.0;
		mask_sup.convertTo(mask_sup, CV_64F, 1.0 / 255.0);
		mask_inf.convertTo(mask_inf, CV_64F, 1.0 / 255.0);
		// Clip them
		seq_temp = seq_temp.mul(1.0 - mask_sup) + mask_sup;
		seq_temp = seq_temp.mul(1.0 - mask_inf);
		// Set to 0 contrast of clipped values
		wc_temp = wc_temp.mul(1.0 - mask_sup);
		wc_temp = wc_temp.mul(1.0 - mask_inf);

		seq[k + Ns] = seq_temp.clone();
		wc[k + Ns] = wc_temp.clone();
	}

	// Compute well-exposedness weights and final normalized weights
	std::vector<cv::Mat> we(N + Ns + 1);
	std::vector<cv::Mat> w(N + Ns + 1);
	cv::Mat sum_w = cv::Mat::zeros(rows, cols, CV_64F);

	for (int i = 0; i < we.size(); i++) {
		cv::Mat p, we_temp, w_temp;
		cv::pow(seq[i] - 0.5, 2, p);
		cv::exp(-0.5*p / (0.2*0.2), we_temp);

		w_temp = wc[i].mul(we_temp);

		we[i] = we_temp.clone();
		w[i] = w_temp.clone();

		sum_w = sum_w + w[i];
	}

	sum_w = 1.0 / sum_w;
	for (int i = 0; i < we.size(); i++) {
		w[i] = w[i].mul(sum_w);
	}

	// Multiscale blending
	cv::Mat lp = multiscale_blending(seq, w);

	if (channels == 1) {
		lp = lp.mul(C);
	}
	else {
		std::vector<cv::Mat> lp3 = { lp.clone(),lp.clone(),lp.clone() };
		cv::merge(lp3, lp);
		lp = lp.mul(C);
	}

	robust_normalization(lp, lp);

	lp.convertTo(dst, CV_8U, 255);

	return;
}