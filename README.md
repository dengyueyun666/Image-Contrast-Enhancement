# Image-Contrast-Enhancement
C++ implementation of "[A New Image Contrast Enhancement Algorithm Using Exposure Fusion Framework](https://baidut.github.io/OpenCE/caip2017.html)".

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
@inproceedings{ying2017new,
  title={A New Image Contrast Enhancement Algorithm Using Exposure Fusion Framework},
  author={Ying, Zhenqiang and Li, Ge and Ren, Yurui and Wang, Ronggang and Wang, Wenmin},
  booktitle={International Conference on Computer Analysis of Images and Patterns},
  pages={36--46},
  year={2017},
  organization={Springer}
}
```
