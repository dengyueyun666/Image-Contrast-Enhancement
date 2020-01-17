// This must be defnined, in order to use arma::spsolve in the code with SuperLU
#define ARMA_USE_SUPERLU

#include <armadillo>
#include <dlib/global_optimization.h>
#include <dlib/optimization.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "util.h"
#include "image_enhancement.h"

void AINDANE(const cv::Mat & src, cv::Mat & dst, int sigma1, int sigma2, int sigma3)
{
	cv::Mat I;
	cv::cvtColor(src, I, CV_BGR2GRAY);

	int histsize = 256;
	float range[] = { 0,256 };
	const float* histRanges = { range };
	int bins = 256;
	cv::Mat hist;
	calcHist(&I, 1, 0, cv::Mat(), hist, 1, &histsize, &histRanges, true, false);

	int L;
	float cdf = 0;
	int total_pixel = src.rows * src.cols;
	for (int i = 0; i < 256; i++) {
		cdf += hist.at<float>(i) / total_pixel;
		if (cdf >= 0.1) {
			L = i;
			break;
		}
	}

	double z;
	if (L <= 50) z = 0;
	else if (L > 150) z = 1;
	else z = (L - 50) / 100.0;

	cv::Mat I_conv1, I_conv2, I_conv3;
	cv::GaussianBlur(I, I_conv1, cv::Size(0, 0), sigma1, sigma1, cv::BORDER_CONSTANT);
	cv::GaussianBlur(I, I_conv2, cv::Size(0, 0), sigma2, sigma2, cv::BORDER_CONSTANT);
	cv::GaussianBlur(I, I_conv3, cv::Size(0, 0), sigma3, sigma3, cv::BORDER_CONSTANT);

	cv::Mat mean, stddev;
	cv::meanStdDev(I, mean, stddev);
	double global_sigma = stddev.at<double>(0, 0);

	double P;
	if (global_sigma <= 3.0)
		P = 3.0;
	else if (global_sigma >= 10.0)
		P = 1.0;
	else
		P = (27.0 - 2.0 * global_sigma) / 7.0;

    // Look-up table.
	uchar Table[256][256];
	for (int Y = 0; Y < 256; Y++)                    // Y represents I_conv(x,y)
	{
		for (int X = 0; X < 256; X++)                // X represents I(x,y)
		{
			double i = X / 255.0;                                                                                   // Eq.2
			i = (std::pow(i, 0.75 * z + 0.25) + (1 - i) * 0.4 * (1 - z) + std::pow(i, 2 - z)) * 0.5;                // Eq.3
			Table[Y][X] = cv::saturate_cast<uchar>(255 * std::pow(i, std::pow((Y + 1.0) / (X + 1.0), P)) + 0.5);    // Eq.7 & Eq.8
		}
	}

	dst = src.clone();
	for (int r = 0; r < src.rows; r++) {
		uchar* I_it = I.ptr<uchar>(r);
		uchar* I_conv1_it = I_conv1.ptr<uchar>(r);
		uchar* I_conv2_it = I_conv2.ptr<uchar>(r);
		uchar* I_conv3_it = I_conv3.ptr<uchar>(r);
		const cv::Vec3b* src_it = src.ptr<cv::Vec3b>(r);
		cv::Vec3b* dst_it = dst.ptr<cv::Vec3b>(r);
		for (int c = 0; c < src.cols; c++) {
			uchar i = I_it[c];
			uchar i_conv1 = I_conv1_it[c];
			uchar i_conv2 = I_conv2_it[c];
			uchar i_conv3 = I_conv3_it[c];
			uchar S1 = Table[i_conv1][i];
			uchar S2 = Table[i_conv2][i];
			uchar S3 = Table[i_conv3][i];
			double S = (S1 + S2 + S3) / 3.0;     // Eq.13

            /***
                The following commented codes are original operation(Eq.14) in paper.
            However, the results may contain obvious color spots due to the difference
            between adjacent enhanced luminance is too large.
            Here is an example:
                original luminance     --->     enhanced luminance
                        1              --->             25
                        2              --->             50
                        3              --->             75
            ***/
			//dst_it[c][0] = cv::saturate_cast<uchar>(src_it[c][0] * S / i);
			//dst_it[c][1] = cv::saturate_cast<uchar>(src_it[c][1] * S / i);
			//dst_it[c][2] = cv::saturate_cast<uchar>(src_it[c][2] * S / i);

            /***
                A simple way to deal with above problem is to limit the amplification,
            says, the amplification should not exceed 4 times. You can adjust it by
            yourself, or adaptively set this value.
                You can uncomment(coment) the above(below) codes to see the difference
            and check it out.
            ***/
			double cof = std::min(S / i, 4.0);
			dst_it[c][0] = cv::saturate_cast<uchar>(src_it[c][0] * cof);
			dst_it[c][1] = cv::saturate_cast<uchar>(src_it[c][1] * cof);
			dst_it[c][2] = cv::saturate_cast<uchar>(src_it[c][2] * cof);
		}
	}
	return;
}


void LDR(const cv::Mat& src, cv::Mat& dst, double alpha)
{
    int R = src.rows;
    int C = src.cols;

    cv::Mat Y;
    std::vector<cv::Mat> YUV_channels;
    if (src.channels() == 1) {
        Y = src.clone();
    } else {
        cv::Mat YUV;
        cv::cvtColor(src, YUV, CV_BGR2YUV);
        cv::split(YUV, YUV_channels);
        Y = YUV_channels[0];
    }

    cv::Mat U = cv::Mat::zeros(255, 255, CV_64F);
    {
        cv::Mat tmp_k(255, 1, CV_64F);
        for (int i = 0; i < 255; i++)
            tmp_k.at<double>(i) = i + 1;

        for (int layer = 1; layer <= 255; layer++) {
            cv::Mat mi, ma;
            cv::min(tmp_k, 256 - layer, mi);
            cv::max(tmp_k - layer, 0, ma);
            cv::Mat m = mi - ma;
            m.copyTo(U.col(layer - 1));
        }
    }

    // unordered 2D histogram acquisition
    cv::Mat h2d = cv::Mat::zeros(256, 256, CV_64F);
    for (int j = 0; j < R; j++) {
        for (int i = 0; i < C; i++) {
            uchar ref = Y.at<uchar>(j, i);

            if (j != R - 1) {
                uchar trg = Y.at<uchar>(j + 1, i);
                h2d.at<double>(std::max(ref, trg), std::min(ref, trg)) += 1;
            }
            if (i != C - 1) {
                uchar trg = Y.at<uchar>(j, i + 1);
                h2d.at<double>(std::max(ref, trg), std::min(ref, trg)) += 1;
            }
        }
    }

    // Intra-Layer Optimization
    cv::Mat D = cv::Mat::zeros(255, 255, CV_64F);
    cv::Mat s = cv::Mat::zeros(255, 1, CV_64F);

    for (int layer = 1; layer <= 255; layer++) {
        cv::Mat h_l = cv::Mat::zeros(256 - layer, 1, CV_64F);

        int tmp_idx = 1;
        for (int j = 1 + layer; j <= 256; j++) {
            int i = j - layer;
            h_l.at<double>(tmp_idx - 1) = std::log(h2d.at<double>(j - 1, i - 1) + 1); // Equation (2)
            tmp_idx++;
        }

        s.at<double>(layer - 1) = cv::sum(h_l)[0];

        if (s.at<double>(layer - 1) == 0)
            continue;

        cv::Mat kernel = cv::Mat::ones(layer, 1, CV_64F);
        cv::Mat m_l = conv2(h_l, kernel, ConvolutionType::CONVOLUTION_FULL); // Equation (30)

        double mi;
        cv::minMaxLoc(m_l, &mi, 0);
        cv::Mat d_l = m_l - mi;
        d_l = d_l.mul(1.0 / U.col(layer - 1)); // Equation (33)

        if (cv::sum(d_l)[0] == 0)
            continue;

        D.col(layer - 1) = d_l / cv::sum(d_l)[0];
    }

    // Inter - Layer Aggregation
    double max_s;
    cv::minMaxLoc(s, 0, &max_s);
    cv::Mat W;
    cv::pow(s / max_s, alpha, W); // Equation (23)
    cv::Mat d = D * W; // Equation (24)

    // reconstruct transformation function
    d /= cv::sum(d)[0];
    cv::Mat tmp = cv::Mat::zeros(256, 1, CV_64F);
    for (int k = 1; k <= 255; k++) {
        tmp.at<double>(k) = tmp.at<double>(k - 1) + d.at<double>(k - 1);
    }
    tmp.convertTo(tmp, CV_8U, 255.0);

    cv::LUT(Y, tmp, Y);

    if (src.channels() == 1) {
        dst = Y.clone();
    } else {
        cv::merge(YUV_channels, dst);
        cv::cvtColor(dst, dst, CV_YUV2BGR);
    }

    return;
}


void Ying_2017_CAIP(const cv::Mat& src, cv::Mat& dst, double mu, double a, double b, double lambda, double sigma)
{
    clock_t start_time, end_time;

    cv::Mat L;
    if (src.channels() == 3) {
        std::vector<cv::Mat> channels;
        split(src, channels);
        L = max(max(channels[0], channels[1]), channels[2]);
    } else {
        L = src.clone();
    }

    cv::Mat normalized_L;
    L.convertTo(normalized_L, CV_64F, 1 / 255.0);

    cv::Mat normalized_half_L;
    resize(normalized_L, normalized_half_L, cv::Size(), 0.5, 0.5, cv::INTER_CUBIC);

    // start_time = clock();
    cv::Mat M_h, M_v;
    {

        cv::Mat dL_h = normalized_half_L.clone();
        normalized_half_L(cv::Range(0, dL_h.rows), cv::Range(1, dL_h.cols)).copyTo(dL_h(cv::Range(0, dL_h.rows), cv::Range(0, dL_h.cols - 1)));
        normalized_half_L(cv::Range(0, dL_h.rows), cv::Range(0, 1)).copyTo(dL_h(cv::Range(0, dL_h.rows), cv::Range(dL_h.cols - 1, dL_h.cols)));
        dL_h = dL_h - normalized_half_L;

        cv::Mat dL_v = normalized_half_L.clone();
        normalized_half_L(cv::Range(1, dL_v.rows), cv::Range(0, dL_v.cols)).copyTo(dL_v(cv::Range(0, dL_v.rows - 1), cv::Range(0, dL_v.cols)));
        normalized_half_L(cv::Range(0, 1), cv::Range(0, dL_v.cols)).copyTo(dL_v(cv::Range(dL_v.rows - 1, dL_v.rows), cv::Range(0, dL_v.cols)));
        dL_v = dL_v - normalized_half_L;

        cv::Mat kernel_h = cv::Mat(1, sigma, CV_64F, cv::Scalar::all(1));
        cv::Mat kernel_v = cv::Mat(sigma, 1, CV_64F, cv::Scalar::all(1));

        cv::Mat gauker_h, gauker_v;
        filter2D(dL_h, gauker_h, -1, kernel_h, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
        filter2D(dL_v, gauker_v, -1, kernel_v, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);

        double sharpness = 0.001;
        M_h = 1.0 / (abs(gauker_h).mul(abs(dL_h)) + sharpness);
        M_v = 1.0 / (abs(gauker_v).mul(abs(dL_v)) + sharpness);
    }
    // end_time = clock();
    // MyTimeOutput("computeTextureWeight处理时间: ", start_time, end_time);

    cv::Mat normalized_T;
    {
        // start_time = clock();
        int r = normalized_half_L.rows;
        int c = normalized_half_L.cols;
        int N = r * c;

        //Since OpenCV data is saved in row-wised,
        //and armadillo data is saved in column-wised,
        //therefore, when we convert OpenCV to Armadillo, we need to do transpose at first.
        cv::Mat M_h_t = M_h.t();
        cv::Mat M_v_t = M_v.t();
        arma::mat wx(reinterpret_cast<double*>(M_h_t.data), r, c);
        arma::mat wy(reinterpret_cast<double*>(M_v_t.data), r, c);

        arma::mat dx = -lambda * arma::reshape(wx, N, 1);
        arma::mat dy = -lambda * arma::reshape(wy, N, 1);

        arma::mat tempx = arma::shift(wx, +1, 1);
        arma::mat tempy = arma::shift(wy, +1, 0);

        arma::mat dxa = -lambda * arma::reshape(tempx, N, 1);
        arma::mat dya = -lambda * arma::reshape(tempy, N, 1);

        tempx.cols(1, c - 1).zeros();
        tempy.rows(1, r - 1).zeros();

        arma::mat dxd1 = -lambda * arma::reshape(tempx, N, 1);
        arma::mat dyd1 = -lambda * arma::reshape(tempy, N, 1);

        wx.col(c - 1).zeros();
        wy.row(r - 1).zeros();

        arma::mat dxd2 = -lambda * arma::reshape(wx, N, 1);
        arma::mat dyd2 = -lambda * arma::reshape(wy, N, 1);

        arma::mat dxd = arma::join_horiz(dxd1, dxd2);
        arma::mat dyd = arma::join_horiz(dyd1, dyd2);

        std::vector<int> x_diag_th = { -N + r, -r };
        std::vector<int> y_diag_th = { -r + 1, -1 };

        arma::sp_mat Ax = spdiags(dxd, x_diag_th, N, N);
        arma::sp_mat Ay = spdiags(dyd, y_diag_th, N, N);

        arma::mat D = 1.0 - (dx + dy + dxa + dya);

        std::vector<int> D_diag_th = { 0 };

        arma::sp_mat A = (Ax + Ay) + (Ax + Ay).t() + spdiags(D, D_diag_th, N, N);

        // end_time = clock();
        // MyTimeOutput("Before Ax=b处理时间: ", start_time, end_time);

        // start_time = clock();

        //Do transpose first.
        cv::Mat normalized_half_L_t = normalized_half_L.t();
        arma::mat normalized_half_L_mat(reinterpret_cast<double*>(normalized_half_L_t.data), r, c);
        arma::vec normalized_half_L_vec = arma::vectorise(normalized_half_L_mat);

        arma::mat t;
        arma::spsolve(t, A, normalized_half_L_vec);
        t.reshape(r, c);

        //When we convert Armadillo to OpenCV, we construct the cv::Mat with row and column number exchange at first.
        //Then do transpose.
        cv::Mat normalized_half_T(c, r, CV_64F, t.memptr());
        normalized_half_T = normalized_half_T.t();

        resize(normalized_half_T, normalized_T, src.size(), 0, 0, CV_INTER_CUBIC);

        // end_time = clock();
        // MyTimeOutput("Ax=b处理时间: ", start_time, end_time);
    }
    // imshow("normalized_T", normalized_T);

    cv::Mat normalized_src;
    src.convertTo(normalized_src, CV_64F, 1 / 255.0);

    // start_time = clock();
    cv::Mat J;
    {

        cv::Mat isBad = normalized_T < 0.5;
        cv::Mat isBad_50x50;
        cv::resize(isBad, isBad_50x50, cv::Size(50, 50), 0, 0, CV_INTER_NN);

        int count = countNonZero(isBad_50x50);
        if (count == 0) {
            J = normalized_src.clone();
        } else {
            isBad_50x50.convertTo(isBad_50x50, CV_64F, 1.0 / 255);

            cv::Mat normalized_src_50x50;
            cv::resize(normalized_src, normalized_src_50x50, cv::Size(50, 50), 0, 0, CV_INTER_CUBIC);
            normalized_src_50x50 = cv::max(normalized_src_50x50, 0);
            cv::Mat Y;
            {
                if (normalized_src_50x50.channels() == 3) {
                    std::vector<cv::Mat> channels;
                    split(normalized_src_50x50, channels);
                    Y = channels[0].mul(channels[1]).mul(channels[2]);
                    cv::pow(Y, 1.0 / 3, Y);
                } else {
                    Y = normalized_src_50x50;
                }
            }
            Y = Y.mul(isBad_50x50);

            dlib::matrix<double> y;
            y.set_size(Y.rows, Y.cols);
            for (int r = 0; r < Y.rows; r++) {
                for (int c = 0; c < Y.cols; c++) {
                    y(r, c) = Y.at<double>(r, c);
                }
            }

            double a = -0.3293, b = 1.1258;

            auto entropy = [&y, &a, &b](double k) {
                double beta = exp(b * (1.0 - pow(k, a)));
                double gamma = pow(k, a);
                double cost = 0;

                std::vector<int> hist(256, 0);
                for (int r = 0; r < y.nr(); r++) {
                    for (int c = 0; c < y.nc(); c++) {
                        double j = beta * pow(y(r, c), gamma);
                        int bin = int(j * 255.0);
                        if (bin < 0)
                            bin = 0;
                        else if (bin >= 255)
                            bin = 255;
                        hist[bin]++;
                    }
                }

                double N = y.nc() * y.nr();

                for (int i = 0; i < hist.size(); i++) {
                    if (hist[i] == 0)
                        continue;
                    double p = hist[i] / N;
                    cost += -p * log2(p);
                }
                return cost;
            };

            auto result = dlib::find_max_global(entropy, 1.0, 7.0, dlib::max_function_calls(20));

            double opt_k = result.x;
            double beta = exp(b * (1.0 - pow(opt_k, a)));
            double gamma = pow(opt_k, a);
            cv::pow(normalized_src, gamma, J);
            J = J * beta - 0.01;

            //cout << "beta: " << beta << endl;
            //cout << "gamma: " << gamma << endl;
            // std::cout << "opt_k: " << opt_k << std::endl;
        }
    }
    // end_time = clock();
    // MyTimeOutput("Ax=J处理时间: ", start_time, end_time);

    cv::Mat T;
    std::vector<cv::Mat> T_channels;
    for (int i = 0; i < src.channels(); i++)
        T_channels.push_back(normalized_T.clone());
    cv::merge(T_channels, T);

    cv::Mat W;
    cv::pow(T, mu, W);

    cv::Mat I2 = normalized_src.mul(W);
    cv::Mat ones_mat = cv::Mat(W.size(), src.channels() == 3 ? CV_64FC3 : CV_64FC1, cv::Scalar::all(1.0));
    cv::Mat J2 = J.mul(ones_mat - W);

    dst = I2 + J2;

    dst.convertTo(dst, CV_8U, 255);

    return;
}


void CEusingLuminanceAdaptation(const cv::Mat& src, cv::Mat& dst)
{
    cv::Mat HSV;
    cv::cvtColor(src, HSV, cv::COLOR_BGR2HSV_FULL);
    std::vector<cv::Mat> HSV_channels;
    cv::split(HSV, HSV_channels);
    cv::Mat V = HSV_channels[2];

    int ksize = 5;
    cv::Mat gauker1 = cv::getGaussianKernel(ksize, 15);
    cv::Mat gauker2 = cv::getGaussianKernel(ksize, 80);
    cv::Mat gauker3 = cv::getGaussianKernel(ksize, 250);

    cv::Mat gauV1, gauV2, gauV3;
    cv::filter2D(V, gauV1, CV_8U, gauker1, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    cv::filter2D(V, gauV2, CV_8U, gauker2, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    cv::filter2D(V, gauV3, CV_8U, gauker3, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

    std::vector<double> lut(256, 0);
    for (int i = 0; i < 256; i++) {
        if (i <= 127)
            lut[i] = 17.0 * (1.0 - std::sqrt(i / 127.0)) + 3.0;
        else
            lut[i] = 3.0 / 128.0 * (i - 127.0) + 3.0;
        lut[i] = (-lut[i] + 20.0) / 17.0;
    }

    cv::Mat beta1, beta2, beta3;
    cv::LUT(gauV1, lut, beta1);
    cv::LUT(gauV2, lut, beta2);
    cv::LUT(gauV3, lut, beta3);

    gauV1.convertTo(gauV1, CV_64F, 1.0 / 255.0);
    gauV2.convertTo(gauV2, CV_64F, 1.0 / 255.0);
    gauV3.convertTo(gauV3, CV_64F, 1.0 / 255.0);

    V.convertTo(V, CV_64F, 1.0 / 255.0);

    cv::log(V, V);
    cv::log(gauV1, gauV1);
    cv::log(gauV2, gauV2);
    cv::log(gauV3, gauV3);

    cv::Mat r = (3.0 * V - beta1.mul(gauV1) - beta2.mul(gauV2) - beta3.mul(gauV3)) / 3.0;

    cv::Mat R;
    cv::exp(r, R);

    double R_min, R_max;
    cv::minMaxLoc(R, &R_min, &R_max);
    cv::Mat V_w = (R - R_min) / (R_max - R_min);

    V_w.convertTo(V_w, CV_8U, 255.0);

    int histsize = 256;
    float range[] = { 0, 256 };
    const float* histRanges = { range };
    cv::Mat hist;
    calcHist(&V_w, 1, 0, cv::Mat(), hist, 1, &histsize, &histRanges, true, false);

    cv::Mat pdf = hist / (src.rows * src.cols);

    double pdf_min, pdf_max;
    cv::minMaxLoc(pdf, &pdf_min, &pdf_max);
    for (int i = 0; i < 256; i++) {
        pdf.at<float>(i) = pdf_max * (pdf.at<float>(i) - pdf_min) / (pdf_max - pdf_min);
    }

    std::vector<double> cdf(256, 0);
    double accum = 0;
    for (int i = 0; i < 255; i++) {
        accum += pdf.at<float>(i);
        cdf[i] = accum;
    }
    cdf[255] = 1.0 - accum;

    double V_w_max;
    cv::minMaxLoc(V_w, 0, &V_w_max);
    for (int i = 0; i < 255; i++) {
        lut[i] = V_w_max * std::pow((i * 1.0 / V_w_max), 1.0 - cdf[i]);
    }

    cv::Mat V_out;
    cv::LUT(V_w, lut, V_out);
    V_out.convertTo(V_out, CV_8U);

    HSV_channels[2] = V_out;
    cv::merge(HSV_channels, HSV);
    cv::cvtColor(HSV, dst, CV_HSV2BGR_FULL);

    return;
}


void adaptiveImageEnhancement(const cv::Mat& src, cv::Mat& dst)
{
    int r = src.rows;
    int c = src.cols;
    int n = r * c;

    cv::Mat HSV;
    cv::cvtColor(src, HSV, cv::COLOR_BGR2HSV_FULL);
    std::vector<cv::Mat> HSV_channels;
    cv::split(HSV, HSV_channels);
    cv::Mat S = HSV_channels[1];
    cv::Mat V = HSV_channels[2];

    int ksize = 5;
    cv::Mat gauker1 = cv::getGaussianKernel(ksize, 15);
    cv::Mat gauker2 = cv::getGaussianKernel(ksize, 80);
    cv::Mat gauker3 = cv::getGaussianKernel(ksize, 250);

    cv::Mat gauV1, gauV2, gauV3;
    cv::filter2D(V, gauV1, CV_64F, gauker1, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    cv::filter2D(V, gauV2, CV_64F, gauker2, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    cv::filter2D(V, gauV3, CV_64F, gauker3, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

    cv::Mat V_g = (gauV1 + gauV2 + gauV3) / 3.0;

    cv::Scalar avg_S = cv::mean(S);
    double k1 = 0.1 * avg_S[0];
    double k2 = avg_S[0];

    cv::Mat V_double;
    V.convertTo(V_double, CV_64F);

    cv::Mat V1 = ((255 + k1) * V_double).mul(1.0 / (cv::max(V_double, V_g) + k1));
    cv::Mat V2 = ((255 + k2) * V_double).mul(1.0 / (cv::max(V_double, V_g) + k2));

    cv::Mat X1 = V1.reshape(0, n);
    cv::Mat X2 = V2.reshape(0, n);

    cv::Mat X(n, 2, CV_64F);
    X1.copyTo(X(cv::Range(0, n), cv::Range(0, 1)));
    X2.copyTo(X(cv::Range(0, n), cv::Range(1, 2)));

    cv::Mat covar, mean;
    cv::calcCovarMatrix(X, covar, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_64F);

    cv::Mat eigenValues; //The eigenvalues are stored in the descending order.
    cv::Mat eigenVectors; //The eigenvectors are stored as subsequent matrix rows.
    cv::eigen(covar, eigenValues, eigenVectors);

    double w1 = eigenVectors.at<double>(0, 0) / (eigenVectors.at<double>(0, 0) + eigenVectors.at<double>(0, 1));
    double w2 = 1 - w1;

    cv::Mat F = w1 * V1 + w2 * V2;

    F.convertTo(F, CV_8U);

    HSV_channels[2] = F;
    cv::merge(HSV_channels, HSV);
    cv::cvtColor(HSV, dst, CV_HSV2BGR_FULL);

    return;
}






