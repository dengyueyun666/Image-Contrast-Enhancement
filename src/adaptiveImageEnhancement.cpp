#include <iostream>
#include <opencv2/opencv.hpp>

#include "image_enhancement.h"

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