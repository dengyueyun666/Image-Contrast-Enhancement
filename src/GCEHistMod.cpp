#include <iostream>
#include <opencv2/opencv.hpp>

#include "image_enhancement.h"

void GCEHistMod(const cv::Mat& src, cv::Mat& dst, int threshold, int b, int w, double alpha, int g)
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
    } else {
        cv::cvtColor(src, HSV, CV_BGR2HSV_FULL);
        cv::split(HSV, HSV_channels);
        L = HSV_channels[2];
    }

    std::vector<int> hist(256, 0);

    int k = 0;
    int count = 0;
    for (int r = 0; r < rows; r++) {
        const uchar* data = L.ptr<uchar>(r);
        for (int c = 0; c < cols; c++) {
            int diff = (c < 2) ? data[c] : std::abs(data[c] - data[c - 2]);
            k += diff;
            if (diff > threshold) {
                hist[data[c]]++;
                count++;
            }
        }
    }

    double kg = k * g;
    double k_prime = kg / std::pow(2, std::ceil(std::log2(kg)));

    double umin = 10;
    double u = std::min(count / 256.0, umin);

    std::vector<double> modified_hist(256, 0);
    double sum = 0;
    for (int i = 0; i < 256; i++) {
        if (i > b && i < w)
            modified_hist[i] = std::round((1 - k_prime) * u + k_prime * hist[i]);
        else
            modified_hist[i] = std::round(((1 - k_prime) * u + k_prime * hist[i]) / (1 + alpha));
        sum += modified_hist[i];
    }

    std::vector<double> CDF(256, 0);
    double culsum = 0;
    for (int i = 0; i < 256; i++) {
        culsum += modified_hist[i] / sum;
        CDF[i] = culsum;
    }

    std::vector<uchar> table_uchar(256, 0);
    for (int i = 1; i < 256; i++) {
        table_uchar[i] = cv::saturate_cast<uchar>(255.0 * CDF[i]);
    }

    cv::LUT(L, table_uchar, L);

    if (channels == 1) {
        dst = L.clone();
    } else {
        cv::merge(HSV_channels, dst);
        cv::cvtColor(dst, dst, CV_HSV2BGR_FULL);
    }

    return;
}