# Image-Contrast-Enhancement
C++ implementation of several image contrast enhancement techniques.

## Techniques
* LDR
  * [Contrast Enhancement based on Layered Difference Representation of 2D Histograms](http://mcl.korea.ac.kr/cwlee_tip2013/) (TIP 2013), Lee et al.
  * Only OpenCV3 is needed.
* Ying_2017_CAIP
  * [A New Image Contrast Enhancement Algorithm Using Exposure Fusion Framework](https://baidut.github.io/OpenCE/caip2017.html) (CAIP 2017), Ying et al.
  * All requirements are needed.
* CEusingLuminanceAdaptation
  * [Retinex-based perceptual contrast enhancement in images using luminance adaptation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8500743) (2018), Fu et al.
  * Only OpenCV3 is needed.
* adaptiveImageEnhancement
  * [Adaptive image enhancement method for correcting low-illumination images](https://www.sciencedirect.com/science/article/pii/S0020025519304104) (2019), Wang et al.
  * Only OpenCV3 is needed.

## Requirements
* Ubuntu-16.04
* Cmake
* OpenCV3
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
```
