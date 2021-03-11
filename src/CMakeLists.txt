#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cmake_minimum_required(VERSION 3.5)
#-------------------------------------------------------------------------------
# cmake project
#-------------------------------------------------------------------------------
project(ALL)

#-------------------------------------------------------------------------------
# Sources
#-------------------------------------------------------------------------------
#
add_subdirectory(2d_3d)
add_subdirectory(camera_model)
add_subdirectory(2view_geometry)
add_subdirectory(correspondence)
add_subdirectory(multiview_geometry)
add_subdirectory(opencv)