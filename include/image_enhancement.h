#ifndef _IMAGE_ENHANCEMENT_H
#define _IMAGE_ENHANCEMENT_H

#include <iostream>
#include <opencv2/opencv.hpp>

/***
@article{tao2005adaptive,
  title={Adaptive and integrated neighborhood-dependent approach for nonlinear enhancement of color images},
  author={Tao, Li and Asari, Vijayan K},
  journal={Journal of Electronic Imaging},
  volume={14},
  number={4},
  pages={043006},
  year={2005},
  publisher={International Society for Optics and Photonics}
}

The code refers to this blog(https://www.cnblogs.com/Imageshop/p/11665100.html).
***/
void AINDANE(const cv::Mat & src, cv::Mat & dst, int sigma1 = 5, int sigma2 = 20, int sigma3 = 120);


/***
@article{wang2007fast,
  title={Fast image/video contrast enhancement based on weighted thresholded histogram equalization},
  author={Wang, Qing and Ward, Rabab K},
  journal={IEEE transactions on Consumer Electronics},
  volume={53},
  number={2},
  pages={757--764},
  year={2007},
  publisher={IEEE}
}
***/
void WTHE(const cv::Mat & src, cv::Mat & dst, float r = 0.5, float v = 0.5);


/***
@article{arici2009histogram,
  title={A histogram modification framework and its application for image contrast enhancement},
  author={Arici, Tarik and Dikbas, Salih and Altunbasak, Yucel},
  journal={IEEE Transactions on image processing},
  volume={18},
  number={9},
  pages={1921--1935},
  year={2009},
  publisher={IEEE}
}
***/
void GCEHistMod(const cv::Mat& src, cv::Mat& dst, int threshold = 5, int b = 23, int w = 230, double alpha = 2, int g = 10);


/***
@article{lee2013contrast,
  title={Contrast enhancement based on layered difference representation of 2D histograms},
  author={Lee, Chulwoo and Lee, Chul and Kim, Chang-Su},
  journal={IEEE transactions on image processing},
  volume={22},
  number={12},
  pages={5372--5384},
  year={2013},
  publisher={IEEE}
}

This is a reimplementation from http://mcl.korea.ac.kr/cwlee_tip2013/
***/
void LDR(const cv::Mat& src, cv::Mat & dst, double alpha = 2.5);


/***
@article{huang2013efficient,
  title={Efficient Contrast Enhancement Using Adaptive Gamma Correction With Weighting Distribution},
  author={Huang, Shihchia and Cheng, Fanchieh and Chiu, Yisheng},
  journal={IEEE Transactions on Image Processing},
  volume={22},
  number={3},
  pages={1032--1041},
  year={2013}
}
***/
void AGCWD(const cv::Mat & src, cv::Mat & dst, double alpha = 0.5);


/***
@article{rahman2016an,
  title={An adaptive gamma correction for image enhancement},
  author={Rahman, Shanto and Rahman, Mostafijur and Abdullahalwadud, M and Alquaderi, Golam Dastegir and Shoyaib, Mohammad},
  journal={Eurasip Journal on Image and Video Processing},
  volume={2016},
  number={1},
  pages={35},
  year={2016}
}
***/
void AGCIE(const cv::Mat & src, cv::Mat & dst);


/***
@article{cao2017contrast,
  title={Contrast enhancement of brightness-distorted images by improved adaptive gamma correction},
  author={Cao, Gang and Huang, Lihui and Tian, Huawei and Huang, Xianglin and Wang, Yongbin and Zhi, Ruicong},
  journal={Computers & Electrical Engineering},
  volume={66},
  pages={569--582},
  year={2017}
}
***/
void IAGCWD(const cv::Mat & src, cv::Mat & dst, double alpha_dimmed = 0.75, double alpha_bright = 0.25, int T_t = 112, double tau_t = 0.3, double tau = 0.5);


/***
@inproceedings{ying2017new,
  title={A New Image Contrast Enhancement Algorithm Using Exposure Fusion Framework},
  author={Ying, Zhenqiang and Li, Ge and Ren, Yurui and Wang, Ronggang and Wang, Wenmin},
  booktitle={International Conference on Computer Analysis of Images and Patterns},
  pages={36--46},
  year={2017},
  organization={Springer}
}

This is a reimplementation from https://baidut.github.io/OpenCE/caip2017.html
***/
void Ying_2017_CAIP(const cv::Mat& src, cv::Mat& dst, double mu = 0.5, double a = -0.3293, double b = 1.1258, double lambda = 0.5, double sigma = 5);


/***
@article{fu2018retinex,
  title={Retinex-based perceptual contrast enhancement in images using luminance adaptation},
  author={Fu, Qingtao and Jung, Cheolkon and Xu, Kaiqiang},
  journal={IEEE Access},
  volume={6},
  pages={61277--61286},
  year={2018},
  publisher={IEEE}
}
***/
void CEusingLuminanceAdaptation(const cv::Mat& src, cv::Mat& dst);



/***
@article{wang2019adaptive,
  title={Adaptive image enhancement method for correcting low-illumination images},
  author={Wang, Wencheng and Chen, Zhenxue and Yuan, Xiaohui and Wu, Xiaojin},
  journal={Information Sciences},
  volume={496},
  pages={25--41},
  year={2019},
  publisher={Elsevier}
}
***/
void adaptiveImageEnhancement(const cv::Mat& src, cv::Mat& dst);


/***
@article{agrawal2019novel,
  title={A novel joint histogram equalization based image contrast enhancement},
  author={Agrawal, Sanjay and Panda, Rutuparna and Mishro, PK and Abraham, Ajith},
  journal={Journal of King Saud University-Computer and Information Sciences},
  year={2019},
  publisher={Elsevier}
}
***/
void JHE(const cv::Mat & src, cv::Mat & dst);


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
void SEF(const cv::Mat & src, cv::Mat & dst, double alpha = 6.0, double beta = 0.5, double lambda = 0.125);

#endif