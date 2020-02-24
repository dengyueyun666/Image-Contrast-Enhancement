#include <iostream>
#include <opencv2/opencv.hpp>

#include "image_enhancement.h"

void AINDANE(const cv::Mat& src, cv::Mat& dst, int sigma1, int sigma2, int sigma3)
{
    cv::Mat I;
    cv::cvtColor(src, I, CV_BGR2GRAY);

    int histsize = 256;
    float range[] = { 0, 256 };
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
    if (L <= 50)
        z = 0;
    else if (L > 150)
        z = 1;
    else
        z = (L - 50) / 100.0;

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
    for (int Y = 0; Y < 256; Y++) // Y represents I_conv(x,y)
    {
        for (int X = 0; X < 256; X++) // X represents I(x,y)
        {
            double i = X / 255.0; // Eq.2
            i = (std::pow(i, 0.75 * z + 0.25) + (1 - i) * 0.4 * (1 - z) + std::pow(i, 2 - z)) * 0.5; // Eq.3
            Table[Y][X] = cv::saturate_cast<uchar>(255 * std::pow(i, std::pow((Y + 1.0) / (X + 1.0), P)) + 0.5); // Eq.7 & Eq.8
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
            double S = (S1 + S2 + S3) / 3.0; // Eq.13

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