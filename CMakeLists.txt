cmake_minimum_required(VERSION 2.8)

project(image_contrast_enhancement)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 3 REQUIRED)
find_package(dlib REQUIRED)
find_package(Armadillo REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ARMADILLO_INCLUDE_DIRS}
    include
)

add_executable(main 
    src/util.cpp
    src/AINDANE.cpp
    src/WTHE.cpp
    src/GCEHistMod.cpp
    src/LDR.cpp
    src/AGCWD.cpp
    src/AGCIE.cpp
    src/IAGCWD.cpp
    src/Ying_2017_CAIP.cpp
    src/CEusingLuminanceAdaptation.cpp
    src/adaptiveImageEnhancement.cpp
    src/JHE.cpp
    src/SEF.cpp
    src/main.cpp
)
target_link_libraries(main  
    ${OpenCV_LIBS} 
    ${ARMADILLO_LIBRARIES}
    dlib::dlib 
)
