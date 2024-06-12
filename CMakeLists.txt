cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE  Debug)
project(AppleDetection)

include(CTest)
enable_testing()

set(OpenCV_DIR "C:/opencv/build") # Updated path

find_package(OPENCV 4.5.5 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} "src/main.cpp")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)