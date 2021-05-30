#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "image_enhancement.h"

using std::chrono::high_resolution_clock;

void MyTimeOutput(const std::string& str, const high_resolution_clock::time_point& start_time, const high_resolution_clock::time_point& end_time)
{
    std::cout << str << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0 << "ms" << std::endl;
    return;
}

int main(int argc, char** argv)
{
    cv::Mat src = cv::imread(argv[1], 1);

    if (src.empty()) {
        std::cout << "Can't read image file." << std::endl;
        return -1;
    }

    high_resolution_clock::time_point start_time, end_time;

    start_time = high_resolution_clock::now();
    cv::Mat AINDANE_dst;
    AINDANE(src, AINDANE_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("AINDANE处理时间: ", start_time, end_time);

    start_time = high_resolution_clock::now();
    cv::Mat WTHE_dst;
    WTHE(src, WTHE_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("WTHE处理时间: ", start_time, end_time);

    start_time = high_resolution_clock::now();
    cv::Mat GCEHistMod_dst;
    GCEHistMod(src, GCEHistMod_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("GCEHistMod处理时间: ", start_time, end_time);

    start_time = high_resolution_clock::now();
    cv::Mat LDR_dst;
    LDR(src, LDR_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("LDR处理时间: ", start_time, end_time);

	start_time = high_resolution_clock::now();
	cv::Mat CLAHE_dst;
	cv::Mat labImage;
	cv::cvtColor(src, labImage, cv::COLOR_BGR2Lab);
	std::vector<cv::Mat> labPlanes(3);
	cv::split(labImage, labPlanes);
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(4);
	cv::Mat dst;
	clahe->apply(labPlanes[0], dst);
	dst.copyTo(labPlanes[0]);
	cv::merge(labPlanes, labImage);
	cv::cvtColor(labImage, CLAHE_dst, cv::COLOR_Lab2BGR);
	end_time = high_resolution_clock::now();
	MyTimeOutput("CLAHE处理时间: ", start_time, end_time);

    start_time = high_resolution_clock::now();
    cv::Mat AGCWD_dst;
    AGCWD(src, AGCWD_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("AGCWD处理时间: ", start_time, end_time);

    start_time = high_resolution_clock::now();
    cv::Mat AGCIE_dst;
    AGCIE(src, AGCIE_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("AGCIE处理时间: ", start_time, end_time);

    start_time = high_resolution_clock::now();
    cv::Mat IAGCWD_dst;
    IAGCWD(src, IAGCWD_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("IAGCWD处理时间: ", start_time, end_time);
	
#ifdef USE_ARMA
    start_time = high_resolution_clock::now();
    cv::Mat Ying_dst;
    Ying_2017_CAIP(src, Ying_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("Ying处理时间: ", start_time, end_time);
#endif
    start_time = high_resolution_clock::now();
    cv::Mat CEusingLuminanceAdaptation_dst;
    CEusingLuminanceAdaptation(src, CEusingLuminanceAdaptation_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("CEusingLuminanceAdaptation处理时间: ", start_time, end_time);

    start_time = high_resolution_clock::now();
    cv::Mat adaptiveImageEnhancement_dst;
    adaptiveImageEnhancement(src, adaptiveImageEnhancement_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("adaptiveImageEnhancement处理时间: ", start_time, end_time);
    
    start_time = high_resolution_clock::now();
    cv::Mat JHE_dst;
    JHE(src, JHE_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("JHE处理时间: ", start_time, end_time);

    start_time = high_resolution_clock::now();
    cv::Mat SEF_dst;
    SEF(src, SEF_dst);
    end_time = high_resolution_clock::now();
    MyTimeOutput("SEF处理时间: ", start_time, end_time);

	cv::namedWindow("src", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::imshow("src", src);
	cv::namedWindow("AINDANE_dst", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::imshow("AINDANE_dst", AINDANE_dst);
	cv::namedWindow("WTHE_dst", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::imshow("WTHE_dst", WTHE_dst);
	cv::namedWindow("GCEHistMod_dst", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::imshow("GCEHistMod_dst", GCEHistMod_dst);
	cv::namedWindow("LDR_dst", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::imshow("LDR_dst", LDR_dst);
	cv::namedWindow("CLAHE_dst", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::imshow("CLAHE_dst", CLAHE_dst);
	cv::namedWindow("AGCWD_dst", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::imshow("AGCWD_dst", AGCWD_dst);
	cv::namedWindow("AGCIE_dst", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::imshow("AGCIE_dst", AGCIE_dst);
	cv::namedWindow("IAGCWD_dst", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::imshow("IAGCWD_dst", IAGCWD_dst);
#ifdef USE_ARMA
	cv::namedWindow("Ying_dst", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::imshow("Ying_dst", Ying_dst);
#endif
	cv::namedWindow("CEusingLuminanceAdaptation_dst", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::imshow("CEusingLuminanceAdaptation_dst", CEusingLuminanceAdaptation_dst);
	cv::namedWindow("adaptiveImageEnhancement_dst", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::imshow("adaptiveImageEnhancement_dst", adaptiveImageEnhancement_dst);
	cv::namedWindow("JHE_dst", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::imshow("JHE_dst", JHE_dst);
	cv::namedWindow("SEF_dst", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::imshow("SEF_dst", SEF_dst);
	
    cv::waitKey();
    return 0;
}