#include <iostream>
#include <opencv2/opencv.hpp>

#include "image_enhancement.h"
#include "util.h"

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