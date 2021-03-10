cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

get_filename_component(INSTALL_DIR "../../bin" REALPATH)

macro(add_app_from_cpp_ceres CPP_NAME)
    find_package(Ceres REQUIRED)
    include_directories(${CERES_INCLUDE_DIRS})
    set(app ${CPP_NAME})
    add_executable(${app} "${CPP_NAME}.cpp")
    target_link_libraries(${app} ${OpenCV_LIBS} ${CERES_LIBRARIES})
    install(TARGETS ${app} DESTINATION ${INSTALL_DIR})
endmacro()

project(correspondence)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories("/usr/local/include/eigen3/")
include_directories("/usr/local/include")

# only need to change the following
add_app_from_cpp_ceres("bundle_adjustment_global")
add_app_from_cpp_ceres("bundle_adjustment_inc")
add_app_from_cpp_ceres("sfm_global")
add_app_from_cpp_ceres("sfm_inc")