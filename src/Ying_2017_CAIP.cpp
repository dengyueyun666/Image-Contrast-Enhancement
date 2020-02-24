// This must be defnined, in order to use arma::spsolve in the code with SuperLU
#define ARMA_USE_SUPERLU

#include <armadillo>
#include <dlib/global_optimization.h>
#include <dlib/optimization.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "image_enhancement.h"
#include "util.h"


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