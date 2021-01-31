cmake_minimum_required(VERSION 2.8)

project(FaceRecognition)

# set(CMAKE_CXX_STANDARD 17)

# OpenCV
find_package(OpenCV 4.5.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp priorbox.cpp utils.cpp FaceRecognition.cpp FaceRecognition.h)
target_link_libraries(main ${OpenCV_LIBS})
