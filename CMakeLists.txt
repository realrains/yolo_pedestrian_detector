cmake_minimum_required(VERSION 3.10)
project(human_detect)

set(CMAKE_CXX_STANDARD 11)

find_package(TensorflowCC REQUIRED COMPONENTS Static)
find_package(OpenCV REQUIRED)

add_executable(human_detect main.cpp YOLO.h)

target_link_libraries(human_detect TensorflowCC::Static)
target_link_libraries(human_detect ${OpenCV_LIBS})