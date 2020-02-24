#include <iostream>
#include <opencv2/opencv.hpp>

#include "image_enhancement.h"

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