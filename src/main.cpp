#include <iostream>
#include <opencv2/opencv.hpp>

#include "image_enhancement.h"

void MyTimeOutput(const std::string& str, const clock_t& start_time, const clock_t& end_time)
{
    std::cout << str << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << std::endl;
    return;
}

void MyTimeOutput(const std::string& str, const clock_t& time)
{
    std::cout << str << (double)(time) / CLOCKS_PER_SEC << "s" << std::endl;
    return;
}


int main(int argc, char** argv)
{
    cv::Mat src = cv::imread(argv[1], 1);

    if (src.empty()) {
        std::cout << "Can't read image file." << std::endl;
        return -1;
    }

    clock_t start_time, end_time;

    start_time = clock();
    cv::Mat AINDANE_dst;
    AINDANE(src, AINDANE_dst);
    end_time = clock();
    MyTimeOutput("AINDANE处理时间: ", start_time, end_time);

    start_time = clock();
    cv::Mat WTHE_dst;
    WTHE(src, WTHE_dst);
    end_time = clock();
    MyTimeOutput("WTHE处理时间: ", start_time, end_time);

    start_time = clock();
    cv::Mat LDR_dst;
    LDR(src, LDR_dst);
    end_time = clock();
    MyTimeOutput("LDR处理时间: ", start_time, end_time);

    start_time = clock();
    cv::Mat AGCWD_dst;
    AGCWD(src, AGCWD_dst);
    end_time = clock();
    MyTimeOutput("AGCWD处理时间: ", start_time, end_time);

    start_time = clock();
    cv::Mat IAGCWD_dst;
    IAGCWD(src, IAGCWD_dst);
    end_time = clock();
    MyTimeOutput("IAGCWD处理时间: ", start_time, end_time);

    start_time = clock();
    cv::Mat Ying_dst;
    Ying_2017_CAIP(src, Ying_dst);
    end_time = clock();
    MyTimeOutput("Ying处理时间: ", start_time, end_time);

    start_time = clock();
    cv::Mat CEusingLuminanceAdaptation_dst;
    CEusingLuminanceAdaptation(src, CEusingLuminanceAdaptation_dst);
    end_time = clock();
    MyTimeOutput("CEusingLuminanceAdaptation处理时间: ", start_time, end_time);

    start_time = clock();
    cv::Mat adaptiveImageEnhancement_dst;
    adaptiveImageEnhancement(src, adaptiveImageEnhancement_dst);
    end_time = clock();
    MyTimeOutput("adaptiveImageEnhancement处理时间: ", start_time, end_time);
    
    start_time = clock();
    cv::Mat JHE_dst;
    JHE(src, JHE_dst);
    end_time = clock();
    MyTimeOutput("JHE处理时间: ", start_time, end_time);

    cv::imshow("src", src);
    cv::imshow("AINDANE_dst", AINDANE_dst);
    cv::imshow("WTHE_dst", WTHE_dst);
    cv::imshow("LDR_dst", LDR_dst);
    cv::imshow("AGCWD_dst", AGCWD_dst);
    cv::imshow("IAGCWD_dst", IAGCWD_dst);
    cv::imshow("Ying_dst", Ying_dst);
    cv::imshow("CEusingLuminanceAdaptation_dst", CEusingLuminanceAdaptation_dst);
    cv::imshow("adaptiveImageEnhancement_dst", adaptiveImageEnhancement_dst);
    cv::imshow("JHE_dst", JHE_dst);
	
    cv::waitKey();
    return 0;
}