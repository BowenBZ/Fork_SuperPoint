cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(SuperPointRunner)

find_package(Torch REQUIRED)
find_package(OpenCV 3.1 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(superpoint_runner superpoint_runner.cpp)
target_link_libraries(superpoint_runner ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET superpoint_runner PROPERTY CXX_STANDARD 17)
