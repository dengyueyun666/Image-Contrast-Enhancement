# Image-Contrast-Enhancement
C++ implementation of several image contrast enhancement techniques.

## Techniques
* AINDANE
  * [Adaptive and integrated neighborhood-dependent approach for nonlinear enhancement of color images](https://www.spiedigitallibrary.org/journals/Journal-of-Electronic-Imaging/volume-14/issue-4/043006/Adaptive-and-integrated-neighborhood-dependent-approach-for-nonlinear-enhancement-of/10.1117/1.2136903.short?SSO=1) (2005), Tao et al.
  * Accepted input image : Color(√) Grayscale(×)
  * Only OpenCV3 is needed.
* WTHE
  * [Fast image/video contrast enhancement based on weighted thresholded histogram equalization](https://ieeexplore.ieee.org/abstract/document/4266969/) (2007), Wang et al.
  * Accepted input image : Color(√) Grayscale(√)
  * Only OpenCV3 is needed.
* GCEHistMod
  * [A histogram modification framework and its application for image contrast enhancement](https://ieeexplore.ieee.org/abstract/document/4895264) (TIP 2009), Arici et al.
  * Accepted input image : Color(√)  Grayscale(√)
  * Only OpenCV3 is needed.
* LDR
  * [Contrast Enhancement based on Layered Difference Representation of 2D Histograms](http://mcl.korea.ac.kr/cwlee_tip2013/) (TIP 2013), Lee et al.
  * Accepted input image : Color(√)  Grayscale(√)
  * Only OpenCV3 is needed.
* AGCWD
  * [Efficient Contrast Enhancement Using Adaptive Gamma Correction With Weighting Distribution](https://ieeexplore.ieee.org/abstract/document/6336819) (TIP 2013), Huang et al.
  * Accepted input image : Color(√)  Grayscale(√)
  * Only OpenCV3 is needed.
* AGCIE
  * [An adaptive gamma correction for image enhancement](https://link.springer.com/article/10.1186%2Fs13640-016-0138-1) (2016), Rahman et al.
  * Accepted input image : Color(√)  Grayscale(√)
  * Only OpenCV3 is needed.
* IAGCWD
  * [Contrast enhancement of brightness-distorted images by improved adaptive gamma correction](https://arxiv.org/abs/1709.04427) (2017), Cao et al.
  * Accepted input image : Color(√)  Grayscale(√)
  * Only OpenCV3 is needed.
* Ying_2017_CAIP
  * [A New Image Contrast Enhancement Algorithm Using Exposure Fusion Framework](https://baidut.github.io/OpenCE/caip2017.html) (CAIP 2017), Ying et al.
  * Accepted input image : Color(√) Grayscale(√)
  * All requirements are needed.
* CEusingLuminanceAdaptation
  * [Retinex-based perceptual contrast enhancement in images using luminance adaptation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8500743) (2018), Fu et al.
  * Accepted input image : Color(√) Grayscale(×)
  * Only OpenCV3 is needed.
* adaptiveImageEnhancement
  * [Adaptive image enhancement method for correcting low-illumination images](https://www.sciencedirect.com/science/article/pii/S0020025519304104) (2019), Wang et al.
  * Accepted input image : Color(√) Grayscale(×)
  * Only OpenCV3 is needed.
* JHE
  * [A novel joint histogram equalization based image contrast enhancement](https://www.sciencedirect.com/science/article/pii/S1319157819303635) (2019), Agrawal et al.
  * Accepted input image : Color(√) Grayscale(√)
  * Only OpenCV3 is needed.
* SEF
  * [An Extended Exposure Fusion and its Application to Single Image Contrast Enhancement](http://openaccess.thecvf.com/content_WACV_2020/papers/Hessel_An_Extended_Exposure_Fusion_and_its_Application_to_Single_Image_WACV_2020_paper.pdf) (WACV 2020), Hessel et al.
  * Accepted input image : Color(√) Grayscale(√)
  * Only OpenCV3 is needed.

## Requirements
* Ubuntu-16.04
* Cmake
* OpenCV-3.4.6+
* Dlib-19.18+
* SuperLU-5.2.1+
* Armadillo-9.800.3+
  * Before install Armadillo, SuperLU 5 must be installed.
  
## Usage
```
cd Image-Contrast-Enhancement
cmake .
make
./main <input_image>
```

## Citations
```
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

@article{huang2013efficient,
  title={Efficient Contrast Enhancement Using Adaptive Gamma Correction With Weighting Distribution},
  author={Huang, Shihchia and Cheng, Fanchieh and Chiu, Yisheng},
  journal={IEEE Transactions on Image Processing},
  volume={22},
  number={3},
  pages={1032--1041},
  year={2013}
}

@article{rahman2016an,
  title={An adaptive gamma correction for image enhancement},
  author={Rahman, Shanto and Rahman, Mostafijur and Abdullahalwadud, M and Alquaderi, Golam Dastegir and Shoyaib, Mohammad},
  journal={Eurasip Journal on Image and Video Processing},
  volume={2016},
  number={1},
  pages={35},
  year={2016}
}

@article{cao2017contrast,
  title={Contrast enhancement of brightness-distorted images by improved adaptive gamma correction},
  author={Cao, Gang and Huang, Lihui and Tian, Huawei and Huang, Xianglin and Wang, Yongbin and Zhi, Ruicong},
  journal={Computers & Electrical Engineering},
  volume={66},
  pages={569--582},
  year={2017}
}

@inproceedings{ying2017new,
  title={A New Image Contrast Enhancement Algorithm Using Exposure Fusion Framework},
  author={Ying, Zhenqiang and Li, Ge and Ren, Yurui and Wang, Ronggang and Wang, Wenmin},
  booktitle={International Conference on Computer Analysis of Images and Patterns},
  pages={36--46},
  year={2017},
  organization={Springer}
}

@article{fu2018retinex,
  title={Retinex-based perceptual contrast enhancement in images using luminance adaptation},
  author={Fu, Qingtao and Jung, Cheolkon and Xu, Kaiqiang},
  journal={IEEE Access},
  volume={6},
  pages={61277--61286},
  year={2018},
  publisher={IEEE}
}

@article{wang2019adaptive,
  title={Adaptive image enhancement method for correcting low-illumination images},
  author={Wang, Wencheng and Chen, Zhenxue and Yuan, Xiaohui and Wu, Xiaojin},
  journal={Information Sciences},
  volume={496},
  pages={25--41},
  year={2019},
  publisher={Elsevier}
}

@article{agrawal2019novel,
  title={A novel joint histogram equalization based image contrast enhancement},
  author={Agrawal, Sanjay and Panda, Rutuparna and Mishro, PK and Abraham, Ajith},
  journal={Journal of King Saud University-Computer and Information Sciences},
  year={2019},
  publisher={Elsevier}
}

@inproceedings{hessel2020extended,
  title={An extended exposure fusion and its application to single image contrast enhancement},
  author={Hessel, Charles and Morel, Jean-Michel},
  booktitle={The IEEE Winter Conference on Applications of Computer Vision},
  pages={137--146},
  year={2020}
}
```

## License
Copyright © 2022 dengyueyun666

This project is under the MIT License. See the LICENSE file for the full license text.
