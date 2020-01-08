cmake_minimum_required(VERSION 2.8)

project(image_contrast_enhancement)

find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)
find_package(Armadillo REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ARMADILLO_INCLUDE_DIRS}
    include
)

add_executable(main 
    src/image_enhancement.cpp
    src/main.cpp
)
target_link_libraries(main  
    ${OpenCV_LIBS} 
    ${ARMADILLO_LIBRARIES}
    dlib::dlib 
)
