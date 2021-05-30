cmake_minimum_required(VERSION 3.8)

project(image_contrast_enhancement)

set(CMAKE_CXX_STANDARD 17)

option(USE_ARMA "Use advanced dependences (arma, dlib)?" OFF)
if (USE_ARMA)
    add_definitions(-DUSE_ARMA)
endif(USE_ARMA)


find_package(OpenCV 4 REQUIRED)
if (USE_ARMA)
   find_package(dlib REQUIRED)
   find_package(Armadillo REQUIRED)
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include
)

set(SOURCES
    src/util.cpp
    src/AINDANE.cpp
    src/WTHE.cpp
    src/GCEHistMod.cpp
    src/LDR.cpp
    src/AGCWD.cpp
    src/AGCIE.cpp
    src/IAGCWD.cpp
    src/CEusingLuminanceAdaptation.cpp
    src/adaptiveImageEnhancement.cpp
    src/JHE.cpp
    src/SEF.cpp
    src/main.cpp
    include/image_enhancement.h
    include/util.h)

set(LIBS ${OpenCV_LIBS})

if (USE_ARMA)
     include_directories(${ARMADILLO_INCLUDE_DIRS})
     set(SOURCES ${SOURCES} src/Ying_2017_CAIP.cpp)
set(LIBS ${LIBS} ${ARMADILLO_LIBRARIES} dlib::dlib)
endif()

add_executable(main ${SOURCES})

target_link_libraries(main ${LIBS})
